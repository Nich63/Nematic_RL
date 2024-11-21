import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt
import h5py

from kinetic_solver import KineticSolver, KineticData
from stable_baselines3 import PPO
from utils.model import DownSampleConv
from utils.defects_gpu import calculate_defects, S_cal, theta_cal
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class CustomPolicyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomPolicyNetwork, self).__init__(observation_space, features_dim)
        
        # 定义网络结构
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.Tanh()  # 使用 tanh 将输出范围限制在 (-1, 1)
        )

    def forward(self, x):
        return self.net(x)

# 定义强化学习环境
class ActiveNematicEnv(gym.Env):
    def __init__(self, solver_paras, seed=1234, random=True, device='cuda:0',
                 solver=None, simulation_data=None, data_path=None, intensity=5,
                 encoder_path=None):
        super(ActiveNematicEnv, self).__init__()
        
        self.device = device
        if solver is None:
            self.solver = KineticSolver(*solver_paras, device=device)

        else:
            self.solver = solver

        if random:
            self.seed = np.random.randint(0, 1000)
        else:
            self.seed = seed

        # (x, y, a, b, theta)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # 32*4*4状态
        self.observation_space = spaces.Box(low=0, high=255, shape=(2, 256, 256), dtype=np.uint8)
        
        if simulation_data is None:
            self.random_flag = True
            # 初始化仿真数据
            self.simulation_data = KineticData(*self.solver.initialize2_pytorch(seed=self.seed), self.solver.simu_args)
            self.simulation_data = self.solver.preloop_kinetic(self.simulation_data, num_itr=10000)
        else:
            self.random_flag = False
            self.simulation_data = simulation_data

        self.data_path = data_path 

        self.intensity = intensity

        self.num_defects = 0

        # self.conv = DownSampleConv().to(device)
        # if encoder_path is not None:
        #     self.conv.load_state_dict(torch.load(encoder_path))
        #     print('Encoder loaded.')
        #     self.conv.eval()
        print('Pre iteration done.')
        
    def reset(self, seed=1234):
        # print('Reseting')
        self.solver.step = 0
        if self.random_flag:
            # 重置仿真数据到初始状态，并返回初始状态
            init_data = self.solver.initialize2_pytorch(seed=self.seed)
            # 更新数据
            self.simulation_data.setter(*init_data)
        else:
            self.simulation_data = self.simulation_data.loader(self.data_path, device=self.device)
        
        return self._convolutional_reduce(), {}

    @staticmethod
    def contains_nan(tuple_of_arrays):
        for i, array in enumerate(tuple_of_arrays):
            if np.isnan(array.cpu().data.numpy()).any():
                print(f"Array {i} contains NaN.")
                return True
        return False

    def step(self, action):
        info = {}
        # print('====steping===')
        # 根据action更新active stress field
        action = np.clip(action, -1, 1)
        light_matrix = self._action2light(self.intensity, self.device, action)
        new_datas, done_flag = self.solver.kinetic(*self.simulation_data.getter(), light_matrix)
        
        # 更新仿真数据
        self.simulation_data.setter(*new_datas)
        
        # 计算奖励和终止条件
        reward = self._calculate_reward()
        terminated = self._check_terminated(done_flag)

        # check if nan in reward
        if self.contains_nan(new_datas):
            print('Nan encontered')
            terminated = True
            info['error'] = "NaN encountered, reset environment."
            reward = -100
            return self.reset(), reward, terminated, info
        
        # 返回下一状态（降维后的）和其他信息
        next_state = self._convolutional_reduce()
        truncated = self._check_truncated()
        return next_state, reward, terminated, truncated, info

    def _check_truncated(self):
        return False

    def render(self, mode='human'):
        # 调用solver的可视化方法
        self.solver.visualize_flow_field(self.simulation_data)

    def _my_render(self, ax):
        # 调用solver的可视化方法
        self.solver.visualize_flow_field(self.simulation_data, ax)

    def close(self):
        # 关闭并清理资源
        self.solver.cleanup()

    def _convolutional_reduce(self):
        # 对数据进行降维
        d_data = self.simulation_data.get_D()
        state = torch.stack(d_data).detach().cpu().numpy()
        # state = state.unsqueeze(0)
        # state = self.conv(state).detach().cpu().numpy()
        state = (state - state.min()) / (state.max() - state.min())
        state = (state * 255).astype(np.uint8)
        return state # here state

    def _calculate_reward(self):
        # reward according S matrix
        d11, d12 = self.simulation_data.get_D()
        
        S = S_cal(d11, d12)
        num_defects = 0
        theta = theta_cal(d11, d12)
        num_defects = len(calculate_defects(theta, grid_size=1, device=self.device))
        self.num_defects = num_defects
        # S is a matrix where zero if defect is present
        # Thus maximize S
        reward = S.mean().item() - 0.1 * self.num_defects
        # print(reward)
        return reward
    
    def _check_terminated(self, done_flag):
        # print('Checking terminated: ', done_flag)
        # if done_flag:
        #     self.solver.step = 0
        # 检查是否达到终止条件
        return done_flag

    @staticmethod
    def _action2light(self_intensity, device, action, grid_size=256):
        action = np.clip(action, -1, 1)
        # 确保 action 在 GPU 上是 PyTorch 张量
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=device)
        
        # 1. 计算椭圆的参数
        x_center = ((action[0] + 1) / 2 * grid_size).int()  # 椭圆平移后的中心x坐标
        y_center = ((action[1] + 1) / 2 * grid_size).int()  # 椭圆平移后的中心y坐标
        a = (action[2] + 1) / 2 * grid_size / 2             # 长轴，假设范围是 [10, 60]
        b = (action[3] + 1) / 2 * grid_size / 2             # 短轴，假设范围是 [5, 35]
        theta = (action[4] + 1) / 2 * 2 * torch.pi          # 方向角范围 [0, 2π]
        intensity = (action[5] + 1) / 2 * self_intensity    # 光照强度范围 [0, intensity]

        # 2. 构建网格坐标，并以中心为原点放置椭圆
        y, x = torch.meshgrid(torch.arange(grid_size, device=device), torch.arange(grid_size, device=device))
        x_shifted = x - grid_size // 2  # 平移椭圆初始位置到正中心
        y_shifted = y - grid_size // 2

        # 3. 旋转椭圆的角度变换
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # 旋转后的坐标
        x_rotated = x_shifted * cos_theta + y_shifted * sin_theta
        y_rotated = -x_shifted * sin_theta + y_shifted * cos_theta

        # 4. 椭圆方程，用于定义椭圆内的区域
        ellipse_mask = (x_rotated / (a + 1e-6))**2 + (y_rotated / (b + 1e-6))**2 <= 1

        # 5. 创建光照矩阵，初始值都设为1，表示椭圆外部区域
        light_matrix = torch.ones((grid_size, grid_size), device=device)

        # 6. 渐变效果 - 根据椭圆边缘距离生成，椭圆内从1到指定强度渐变
        distance = torch.sqrt((x_rotated / (a + 1e-6))**2 + (y_rotated / (b + 1e-6))**2)
        light_matrix_centered = torch.where(ellipse_mask, intensity - (intensity - 1) * distance, light_matrix)

        # 7. 将光照矩阵平移到指定的中心位置
        x_translation = (x_center - grid_size // 2) % grid_size
        y_translation = (y_center - grid_size // 2) % grid_size

        # 8. 使用 roll 函数将矩阵平移并处理周期性边界条件
        light_matrix_shifted = torch.roll(light_matrix_centered, shifts=(y_translation.item(), x_translation.item()), dims=(0, 1))
        # light_matrix_shifted = torch.ones((grid_size, grid_size), device=device)
        return light_matrix_shifted


from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class MyCallback(BaseCallback):
    """
    自定义回调，用于记录每一步的奖励并保存到 TensorBoard
    """
    # TODO: 断点重续
    def __init__(self, verbose=1,
                name_prefix: str = 'rl_model',
                save_freq=2000,
                plot_freq=5000,
                env=None,
                total_timesteps=20000,
                dump_flag=False):
        super().__init__(verbose)
        
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.env = env.envs[0]
        self.writer = None
        self.default_log_path = "/home/hou63/pj2/Nematic_RL/logs_2/default"

        self.plot_freq = plot_freq
        self.interval = self.env.solver.inner_steps
        # self.interval = env.get_attr('solver').inner_steps
        self.plot_flag = False
        self.dump_flag = dump_flag

        self.step_counter = 0
        self.plot_counter = 0
        self.num_plot = 10
        self.total_timesteps = total_timesteps
        if dump_flag:
            self.action_dump = np.zeros((total_timesteps+1, 6))
            self.d_dump = torch.zeros((total_timesteps//10+1, 2, 256, 256)).to(self.env.device)
    
        # self.prog_bar = None

    def _on_training_start(self) -> None:
        if self.logger.dir is not None:
            print("Logger directory set. Using logger directory.")
            self.writer = SummaryWriter(self.logger.dir)
        else:
            print("Logger directory not set. Using default log directory.")
            self.writer = SummaryWriter(self.default_log_path)
        # self.progress_bar = tqdm(total=self.total_timesteps, dynamic_ncols=True, leave=False)

    def _on_rollout_end(self) -> None:
        # 获取 rollout 中的所有奖励
        rewards = self.locals['rollout_buffer'].rewards
        # 计算平均奖励
        average_reward = rewards.mean()
        # 记录平均奖励到日志中
        self.logger.record("rollout/average_reward", average_reward)
        
        # 打印信息（可选）
        if self.verbose > 0:
            print(f"Rollout average reward: {average_reward}")

    def _on_step(self) -> bool:
        """
        每一步环境交互时调用
        """

        # self.progress_bar.update(1)
        # self.progress_bar.refresh() 

        # get reward and action at step
        reward = self.locals['rewards']
        reward = reward[0]
        action = self.locals['actions'][0]

        # if np.any(action < -1) or np.any(action > 1):
        #     print(f"Action out of bounds! Action: {action}, Expected: {-1} to {1}")

        # 记录当前步骤的奖励
        if self.n_calls <= 100000:
            self.writer.add_scalar("step_single/reward", reward, global_step=self.num_timesteps)
        # self.logger.record("value/random", value)
        self.writer.add_scalar("step_single/num_defects", self.env.num_defects, global_step=self.num_timesteps)
    
        # save model
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(self.logger.dir, f"{self.name_prefix}_{self.num_timesteps}.zip")
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {checkpoint_path} at step {self.num_timesteps}")

        # print into console
        if self.n_calls % 1000 == 0:
            print(f"Step {self.num_timesteps}/{self.total_timesteps} - "
                f"Progress: {self.num_timesteps / self.total_timesteps:.2%}", flush=True)

        if self.dump_flag:
            self.action_dump[self.num_timesteps-1] = action
            if self.num_timesteps % 10 == 0:
                # TODO: check if nan in D
                d11, d12 =  self.env.simulation_data.get_D()
                self.d_dump[self.num_timesteps//10] = torch.stack([d11, d12])


        if self.dump_flag and self.n_calls==self.total_timesteps:
            h5_path = os.path.join(self.logger.dir, 'data_dump.h5')
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('actions', data=self.action_dump)
                f.create_dataset('D', data=self.d_dump.cpu().numpy())
            print('Data dump saved to ', h5_path)

        # plot
        if self.n_calls % self.plot_freq == 0 or self.n_calls == 1:
            self.plot_flag = True
        if self.plot_flag:
            if self.step_counter % self.interval == 0:
                fig = self.plot()
                self.writer.add_figure('flow_field', fig, global_step=self.num_timesteps)
                plt.close(fig)
                print('Plotting t = ', self.num_timesteps)
                self.plot_counter += 1
            self.step_counter += 1

        if self.plot_counter >= self.num_plot:
            self.plot_flag = False
            self.plot_counter = 0
            self.step_counter = 0
        
        return True

    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        self.env._my_render(ax)
        light_mat = self.env._action2light(self.env.intensity, self.env.device, self.locals['actions'][0]).cpu().numpy()
        fig.colorbar(ax.imshow(light_mat,
                                cmap='gray', alpha=0.5, vmin=0, vmax=self.env.intensity))
        ax.set_title(str(self.locals['actions'][0]), fontsize=6)
        ax.set_xlabel('num of defects: ' + str(self.env.num_defects))
        # plt.colorbar()
        return fig

    def _on_training_end(self) -> None:
        # 关闭进度条
        # if self.progress_bar is not None:
        #     self.progress_bar.close()

        # 关闭 TensorBoard 日志
        self.writer.close()
        if self.dump_flag:
            h5_path = os.path.join(self.logger.dir, 'data_dump_1.h5')
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('actions', data=self.action_dump)
                f.create_dataset('D', data=self.d_dump.cpu().numpy())
            print('In the End. Data dump saved to ', h5_path)


if __name__ == '__main__':
    # 初始化物理仿真器
    geo_params = {
        'N': 256,
        'Nth': 256,
        'L': 20
    }
    flow_params = {
        'dT': 0.3,
        'dR': 0.3,
        'alpha': -10,
        'beta': 1.0,
        'zeta': 2,
        'V0': 0.0
    }
    simu_params = {
        'dt': 0.0004,
        'seed': 1234,
        'inner_steps': 8,
        'outer_steps': 16
    }
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    solver_paras = (geo_params, flow_params, simu_params)
    # env = ActiveNematicEnv(solver_paras, device=device)

    # solver_paras = (geo_params, flow_params, simu_params)

    solver = KineticSolver(*solver_paras, device=device)
    data_path = '/home/hou63/pj2/Nematic_RL/datas/simulation_data_test.pkl'
    # simulation_data = KineticData(*solver.initialize2_pytorch(seed=918), solver.simu_args)
    simulation_data = KineticData.loader(data_path)
    # simulation_data = solver.preloop_kinetic(simulation_data, num_itr=32000)
    print(simulation_data)
    simulation_data = simulation_data.loader(data_path)
    env = ActiveNematicEnv(solver_paras, solver=solver,
                            simulation_data=simulation_data, device=device,
                            data_path=data_path, intensity=10)
    model = PPO(
    ActorCriticCnnPolicy, env, verbose=1, device=device,
    n_steps=4, batch_size=2,
    tensorboard_log="/home/hou63/pj2/Nematic_RL/logs")

    tic = time.time()
    model.learn(total_timesteps=24, callback=MyCallback())
    toc = time.time()
    print('Time cost: ', toc - tic)