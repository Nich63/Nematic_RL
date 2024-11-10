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
import imageio

from utils.model import DownSampleConv
from utils.defects import theta_cal, S_cal
from kinetic_solver import KineticSolver, KineticData
from nematic_env import ActiveNematicEnv, MyCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.monitor import ResultsWriter


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
data_path = '/home/yuh113/pj2/Nematic_RL/datas/data_2000.pkl'
# data_path = '/home/hou63/pj2/Nematic_RL/datas/simulation_data_test.pkl'
# simulation_data = KineticData(*solver.initialize2_pytorch(seed=918), solver.simu_args)
simulation_data = KineticData.loader(data_path)

encoder_path = '/home/yuh113/pj2/Nematic_RL/log_model/encoder_checkpoint.pth'  # 模型保存路径


# simulation_data = solver.preloop_kinetic(simulation_data, num_itr=32000)
print(simulation_data)
simulation_data = simulation_data.loader(data_path, device=device)
env = ActiveNematicEnv(solver_paras, solver=solver,
                        simulation_data=simulation_data, device=device,
                        data_path=data_path, intensity=10, encoder_path=encoder_path)

modal_path = '/home/yuh113/pj2/Nematic_RL/logs_ucsd/PPO_11/lights_on_model.zip'
model = PPO.load(modal_path, env=env)

print('model loaded')