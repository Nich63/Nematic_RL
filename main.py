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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from utils.model import DownSampleConv
from utils.defects import theta_cal, S_cal
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from kinetic_solver import KineticSolver, KineticData
from nematic_env import ActiveNematicEnv, MyCallback


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

solver_paras = (geo_params, flow_params, simu_params)
# env = ActiveNematicEnv(solver_paras, device=device)

# solver_paras = (geo_params, flow_params, simu_params)

solver = KineticSolver(*solver_paras, device=device)
data_path = '/home/hou63/pj2/Nematic_RL/datas/data_2000.pkl'
# data_path = '/home/hou63/pj2/Nematic_RL/datas/simulation_data_test.pkl'
# simulation_data = KineticData(*solver.initialize2_pytorch(seed=918), solver.simu_args)
simulation_data = KineticData.loader(data_path)

encoder_path = '/home/hou63/pj2/Nematic_RL/log_model/encoder_checkpoint.pth'  # 模型保存路径



# simulation_data = solver.preloop_kinetic(simulation_data, num_itr=32000)
print(simulation_data)
simulation_data = simulation_data.loader(data_path, device=device)
env = ActiveNematicEnv(solver_paras, solver=solver,
                        simulation_data=simulation_data, device=device,
                        data_path=data_path, intensity=10, encoder_path=encoder_path)

# check_env(env)
# # 使用PPO算法进行强化学习
model = PPO(
    ActorCriticCnnPolicy, env, verbose=1, device=device,
    n_steps=2500, batch_size=100,
    tensorboard_log="/home/hou63/pj2/Nematic_RL/logs_3")

name_prefix = 'lights_on_model'

# model_path = '/home/hou63/pj2/Nematic_RL/logs_2/PPO_3/lights_on_model_20000_steps.zip'
# model.load(model_path)

# tic = time.time()
model.learn(total_timesteps=15000,
            callback=MyCallback(
                name_prefix=name_prefix,
                save_freq=5000,
                plot_freq=2500,
                env=env),
            progress_bar=True)
# toc = time.time()
# print('Time cost: ', toc - tic)