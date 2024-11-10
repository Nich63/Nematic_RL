import torch
import matplotlib.pyplot as plt
from utils.defects import S_cal
import numpy as np
import pickle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import tensorboard
import time

from utils.model import DownSampleConv
from utils.defects import theta_cal, S_cal
from kinetic_solver import KineticSolver, KineticData
from nematic_env import ActiveNematicEnv, MyCallback, CustomPolicyNetwork

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.monitor import ResultsWriter
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# 初始化物理仿真器
geo_params = {
    'N': 256,
    'Nth': 256,
    'L': 10
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
    'inner_steps': 10,
    'outer_steps': 5000
}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device:', device)

solver_paras = (geo_params, flow_params, simu_params)
# env = ActiveNematicEnv(solver_paras, device=device)

# solver_paras = (geo_params, flow_params, simu_params)

solver = KineticSolver(*solver_paras, device=device)
data_path = '/home/yuh113/pj2/Nematic_RL/datas/data_2000.pkl'
# data_path = '/home/hou63/pj2/Nematic_RL/datas/simulation_data_test.pkl'
# simulation_data = KineticData(*solver.initialize2_pytorch(seed=918), solver.simu_args)
simulation_data = KineticData.loader(data_path)
# tmp_data = simulation_data.copy()
# tmp_data = tmp_data.to('cpu')
# # save this to local
# with open('/home/yuh113/pj2/Nematic_RL/datas/tmp_data.pkl', 'wb') as f:
#     pickle.dump(tmp_data, f)
# print('tmp_data saved')

encoder_path = '/home/yuh113/pj2/Nematic_RL/log_model/encoder_checkpoint.pth'  # 模型保存路径


# simulation_data = solver.preloop_kinetic(simulation_data, num_itr=32000)
print(simulation_data)
simulation_data = simulation_data.loader(data_path, device=device)
myenv = ActiveNematicEnv(solver_paras, solver=solver,
                        simulation_data=simulation_data, device=device,
                        data_path=data_path, intensity=10, encoder_path=encoder_path)

env = DummyVecEnv([lambda: myenv])
env = VecNormalize(env, training=True, norm_obs=True, norm_reward=False)

# check_env(env)
# # # 使用PPO算法进行强化学习



###############################################################
policy_kwargs = dict(features_extractor_class=CustomPolicyNetwork)

model = PPO(
    ActorCriticCnnPolicy, env, verbose=1, device=device,
    n_steps=5000, batch_size=200,
    tensorboard_log="/home/yuh113/pj2/Nematic_RL/logs_ucsd",
    n_epochs=4, clip_range=0.2, gae_lambda=0.95,
    policy_kwargs=dict(normalize_images=False))

name_prefix = 'lights_on_model'

model_path = '/home/yuh113/pj2/Nematic_RL/logs_ucsd/PPO_15/lights_on_model.zip'
model.load(model_path)

# tic = time.time()
model.learn(total_timesteps=25000,
            callback=MyCallback(
                name_prefix=name_prefix,
                save_freq=5000,
                plot_freq=5000,
                env=env),
            progress_bar=True)
# toc = time.time()
# print('Time cost: ', toc - tic)