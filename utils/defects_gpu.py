import numpy as np
import torch
import scipy.io as sio
import time

from kinetic_solver import KineticData

def S_cal(d11, d12):
    q11 = d11 - 0.5
    q12 = d12
    S = 2 * torch.sqrt(q11**2 + q12**2)
    return S

def theta_cal(d11, d12):
    return 0.5 * torch.atan2(2*d12, 2*d11-1)

def delta_theta_cal(theta1, theta2):
    delta = theta2 - theta1
    delta = torch.where(delta < -torch.pi / 2, delta + torch.pi, delta)
    delta = torch.where(delta > torch.pi / 2, delta - torch.pi, delta)
    return delta

def calculate_defects(theta_field, grid_size=1, device="cuda"):
    rows, cols = theta_field.shape

    # 定义defect list
    defects = []

    # 提取小区域的索引坐标
    i_indices = torch.arange(grid_size, rows - grid_size, grid_size, device=device)
    j_indices = torch.arange(grid_size, cols - grid_size, grid_size, device=device)
    
    # 使用索引网格来构建所有小区域的中心点
    ii, jj = torch.meshgrid(i_indices, j_indices, indexing='ij')
    
    # 获取每个小区域的角度值
    theta_values = torch.stack([
        theta_field[ii - 1, jj],     # 上
        theta_field[ii - 1, jj + 1], # 上右
        theta_field[ii, jj + 1],     # 右
        theta_field[ii + 1, jj + 1], # 下右
        theta_field[ii + 1, jj],     # 下
        theta_field[ii + 1, jj - 1], # 下左
        theta_field[ii, jj - 1],     # 左
        theta_field[ii - 1, jj - 1], # 上左
        theta_field[ii - 1, jj]      # 上 (闭环)
    ], dim=0)

    # 计算每个小区域的 delta_theta
    delta_thetas = delta_theta_cal(theta_values[:-1], theta_values[1:])
    delta_theta_sums = delta_thetas.sum(dim=0)  # 求和得到总的 delta_theta
    
    # 判断哪些区域符合缺陷条件
    is_positive_defect = torch.abs(delta_theta_sums - torch.pi) < 0.5
    is_negative_defect = torch.abs(delta_theta_sums + torch.pi) < 0.5

    # 获取缺陷的坐标并存储
    positive_defect_indices = torch.nonzero(is_positive_defect, as_tuple=True)
    negative_defect_indices = torch.nonzero(is_negative_defect, as_tuple=True)
    
    for i, j in zip(ii[positive_defect_indices], jj[positive_defect_indices]):
        defects.append((i.item(), j.item(), 0.5))
    for i, j in zip(ii[negative_defect_indices], jj[negative_defect_indices]):
        defects.append((i.item(), j.item(), -0.5))
    
    return defects