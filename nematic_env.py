import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt

from kinetic_solver import KineticSolver, KineticData
from stable_baselines3 import PPO
from utils.model import DownSampleConv
from utils.defects import theta_cal, S_cal
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from torch.utils.tensorboard import SummaryWriter


# 定义强化学习环境
class ActiveNematicEnv(gym.Env):
    def __init__(self, solver_paras, seed=1234, random=True, device='cuda:0',
                 solver=None, simulation_data=None, data_path=None, intensity=5):
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
        
        # 定义动作空间和状态空间
        # 假设动作是 6 维向量 (x, y, a, b, theta)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # 假设状态空间是卷积降维后的32*4*4状态
        self.observation_space = spaces.Box(low=0, high=255, shape=(16, 64, 64), dtype=np.uint8)
        
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

        self.conv = DownSampleConv().to(device)
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
        
        # 返回降维后的初始状态
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
        light_matrix = self._action2light(self.intensity, action)
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
            info['error'] = "NaN encountered"
            reward = -100
        
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
        state = torch.stack(d_data)
        state = state.unsqueeze(0)
        state = self.conv(state).detach().cpu().numpy()
        state = (state - state.min()) / (state.max() - state.min())
        state = (state * 255).astype(np.uint8)
        return state # here state

    def _calculate_reward(self):
        # reward according S matrix
        d11, d12 = self.simulation_data.get_D()
        
        S = S_cal(d11.cpu().data.numpy(), d12.cpu().data.numpy())
        # S is a matrix where zero if defect is present
        # Thus maximize S
        reward = S.mean().item()
        # print(reward)
        return reward
    
    def _check_terminated(self, done_flag):
        # print('Checking terminated: ', done_flag)
        # if done_flag:
        #     self.solver.step = 0
        # 检查是否达到终止条件
        return done_flag

    @staticmethod
    def _action2light(self_intensity, action, grid_size=256):

        # 1. 计算椭圆的参数
        x_center = int((action[0] + 1) / 2 * grid_size)  # 椭圆平移后的中心x坐标
        y_center = int((action[1] + 1) / 2 * grid_size)  # 椭圆平移后的中心y坐标
        a = (action[2] + 1) / 2 * grid_size / 2          # 长轴，假设范围是 [10, 60]
        b = (action[3] + 1) / 2 * grid_size / 2          # 短轴，假设范围是 [5, 35]
        theta = (action[4] + 1) / 2 * 2 * np.pi          # 方向角范围 [0, 2π]
        intensity = (action[5] + 1) / 2 * self_intensity     # 光照强度范围 [0, intensity]

        # 2. 构建网格坐标，首先以中心为原点放置椭圆
        y, x = np.ogrid[:grid_size, :grid_size]
        x_shifted = x - grid_size // 2  # 椭圆初始位置在中心，平移到正中间
        y_shifted = y - grid_size // 2

        # 3. 旋转椭圆的角度变换
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 旋转后的坐标
        x_rotated = x_shifted * cos_theta + y_shifted * sin_theta
        y_rotated = -x_shifted * sin_theta + y_shifted * cos_theta
        
        # 4. 椭圆方程 (x_rotated / a)^2 + (y_rotated / b)^2 <= 1 用于定义椭圆内的区域
        ellipse_mask = (x_rotated / (a + 1e-6))**2 + (y_rotated / (b + 1e-6))**2 <= 1

        # 5. 创建光照矩阵，初始值都设为1，表示椭圆外部区域
        light_matrix = np.ones((grid_size, grid_size))

        # 6. 渐变效果 - 根据椭圆边缘距离生成，椭圆内从1到指定强度渐变
        distance = np.sqrt((x_rotated / (a + 1e-6))**2 + (y_rotated / (b + 1e-6))**2)
        light_matrix_centered = np.where(ellipse_mask, intensity - (intensity - 1) * distance, light_matrix)

        # 7. 将光照矩阵平移到指定的中心位置
        x_translation = (x_center - grid_size // 2) % grid_size
        y_translation = (y_center - grid_size // 2) % grid_size

        # 8. 使用 roll 函数将矩阵平移并处理周期性边界条件
        light_matrix_shifted = np.roll(light_matrix_centered, shift=(y_translation, x_translation), axis=(0, 1))
        # light_matrix_shifted = np.ones((grid_size, grid_size))
        return light_matrix_shifted

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class MyCallback(BaseCallback):
    """
    自定义回调，用于记录每一步的奖励并保存到 TensorBoard
    """
    # TODO: 断点重续
    def __init__(self, verbose=0,
                name_prefix: str = 'rl_model',
                save_freq=2000,
                plt_freq=5000,
                env=None):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.plt_freq = plt_freq
        self.env = env
        self.writer = None
        self.default_log_path = "/home/hou63/pj2/Nematic_RL/logs_2/default"
        self.interval = env.solver.inner_steps
        self.plot_flag = False
        self.step_counter = 0
        self.plot_counter = 0
        self.num_plot = 8

    def _on_training_start(self) -> None:
        if self.logger.dir is not None:
            print("Logger directory set. Using logger directory.")
            self.writer = SummaryWriter(self.logger.dir)
        else:
            print("Logger directory not set. Using default log directory.")
            self.writer = SummaryWriter(self.default_log_path)

    def _on_step(self) -> bool:
        """
        每一步环境交互时调用
        """
        reward = self.locals['rewards']  # 从环境中获取当前的奖励
        reward = reward[0]
        # 记录当前步骤的奖励
        # value = np.random.random()
        self.logger.record("step_single/rewards", reward)
        # self.logger.record("value/random", value)
    
        # save model
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(self.logger.dir, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {checkpoint_path} at step {self.num_timesteps}")

        # # plot and save
        # if self.n_calls % self.plt_freq == 0 or self.n_calls == 1:
        #     if self.step_counter % self.interval == 0:
        #         fig = self.plot()
        #         self.writer.add_figure('flow_field', fig, global_step=self.num_timesteps)
        #         plt.close(fig)
        #     self.step_counter += 1

        # plot
        if self.n_calls % self.plt_freq == 0 or self.n_calls == 1:
            self.plot_flag = True
        if self.plot_flag:
            if self.step_counter % self.interval == 0:
                fig = self.plot()
                self.writer.add_figure('flow_field', fig, global_step=self.num_timesteps)
                plt.close(fig)
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
        ax.imshow(
            self.env._action2light(self.env.intensity,
                                   self.locals['actions'][0]),
                                   cmap='gray', alpha=0.5)
        return fig


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