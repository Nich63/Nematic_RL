import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import sobel


def delta_theta_cal(theta1, theta2):
    delta = theta2 - theta1
    if delta < -np.pi/2:
        delta = delta + np.pi
    if delta > np.pi/2:
        delta = delta - np.pi
    return delta

def calculate_defects(theta_field, grid_size=1):
        defects = []
        rows, cols = theta_field.shape
        for i in range(grid_size, rows - grid_size, grid_size):
            for j in range(grid_size, cols - grid_size, grid_size):
                # 在小的区域内计算角度变化，考虑角度跳变问题
                delta_theta = (delta_theta_cal(theta_field[i-1, j], theta_field[i-1, j+1]) +
                                delta_theta_cal(theta_field[i-1, j+1], theta_field[i, j+1]) +
                                delta_theta_cal(theta_field[i, j+1], theta_field[i+1, j+1]) +
                                delta_theta_cal(theta_field[i+1, j+1], theta_field[i+1, j]) +
                                delta_theta_cal(theta_field[i+1, j], theta_field[i+1, j-1]) +
                                delta_theta_cal(theta_field[i+1, j-1], theta_field[i, j-1]) +
                                delta_theta_cal(theta_field[i, j-1], theta_field[i-1, j-1]) +
                                delta_theta_cal(theta_field[i-1, j-1], theta_field[i-1, j]))
                # 使用模运算确保角度连续
                # delta_theta = np.mod(delta_theta + np.pi, 2 * np.pi) - np.pi
                # delta_theta = np.mod(delta_theta, np.pi)
                
                # 通过角度变化计算缺陷的拓扑电荷
                if np.abs(delta_theta - np.pi) < 0.5:
                    defect_type = 0.5
                    defects.append((i, j, defect_type))

                if np.abs(delta_theta + np.pi) < 0.5:
                    defect_type = -0.5
                    defects.append((i, j, defect_type))
        return defects

def plot_defects(defects, theta, ax0):
    cost = np.cos(theta)
    sint = np.sin(theta)
    ax0.streamplot(np.arange(0, theta.shape[0]), np.arange(0, theta.shape[1]),
                    cost, sint, arrowsize=0.5, color='r',
                    density=4, linewidth=0.5)
    ax0.set_aspect('equal')
    for defect in defects:
        i, j, defect_type = defect
        color = 'r' if defect_type > 0 else 'b'
        ax0.scatter(j, i, color=color)
    return ax0

def find_neighboor(ind, pts, eps, shape):
    shape_0, shape_1 = shape
    center_0, center_1 = shape_0//2, shape_1//2
    pt_tmp = pts[ind]
    dist_to_center = np.array([center_0, center_1]) - pt_tmp
    pts_moved = pts + dist_to_center
    pts_moved = pts_moved % np.array([shape_0, shape_1])
    pt_tmp = pts_moved[ind]
    dist = np.linalg.norm(pts_moved - pt_tmp, axis=1)
    neighboor = np.where(dist < eps)
    return neighboor

def find_clusters(flag,eps=3, min_pts=20):
    pt0, pt1 = np.where(flag)
    pts = np.array([pt0, pt1]).T
    neighboor_list = []
    central_list = []
    for i in range(pts.shape[0]):
        neighboor = find_neighboor(i, pts, eps, flag.shape)
        if neighboor[0].shape[0] > min_pts:
            neighboor_list.append(neighboor[0])
            central_list.append(i)
    central_list = np.array(central_list)
    clusters = []  # 存储最终的簇
    not_visited = list(range(len(central_list)))  # 用于跟踪未访问的点

    while not_visited:
        # 从未访问的点中取出第一个核心点
        first_index = not_visited[0]
        cluster = set(neighboor_list[first_index])  # 使用集合来存储簇

        not_visited.remove(first_index)

        # 进行合并
        while True:
            merged = False
            for i in range(len(central_list)):
                ind = central_list[i]
                if i in not_visited and ind in cluster:  # 核心点在集群中
                    not_visited.remove(i)
                    # 只合并当前核心点的邻居
                    new_neighbors = set(neighboor_list[i])
                    if not new_neighbors.issubset(cluster):  # 确保不重复合并
                        cluster.update(new_neighbors)  # 合并邻居
                        merged = True
            if not merged:  # 如果没有新的合并，跳出循环
                break

        clusters.append(np.array(list(cluster)))  # 转换为数组并添加到簇列表中
    

    
    # turn clusters from index to coordinates
    clusters_coord = []
    for cluster in clusters:
        cluster_coord = pts[cluster]
        clusters_coord.append(cluster_coord)

    return clusters_coord

def find_defects_in_clusters(clusters_coord, defects):
    defects_new = []
    for cluster in clusters_coord:
        defects_in_cluster = []
        defect_info = []
        for defect in defects:
            if np.any(np.all(cluster == defect[0:2], axis=1)):
                defect_info.append(defect)
        if defect_info:
            defects_new.append((cluster, defect_info))

    return defects_new

def defects_analysis(theta_field, flag, grid_size=1, eps=3, min_pts=20):
    defects = calculate_defects(theta_field, grid_size)
    clusters_coord = find_clusters(flag, eps, min_pts)
    defects_new = find_defects_in_clusters(clusters_coord, defects)
    return defects_new

def defects_plot(defects_new, theta, ax0, ax1):
    # fig, axs = plt.subplots(1, 2, figsize=(14,14))
    ax0.imshow(theta, cmap='viridis', origin='lower')
    for cluster, defects in defects_new:
        cluster = np.array(cluster)
        ax0.scatter(cluster[:, 1], cluster[:, 0], color='g', s=5)
        for defect in defects:
            i, j, defect_type = defect
            color = 'r' if defect_type > 0 else 'b'
            ax0.scatter(j, i, color=color)
    cost, sint = np.cos(theta), np.sin(theta)
    ax1.streamplot(np.arange(0, theta.shape[0]), np.arange(0, theta.shape[1]),
                    cost, sint, arrowsize=0.5, color='r',
                    density=4, linewidth=0.5)
    for cluster, defects in defects_new:
        cluster = np.array(cluster)
        ax1.scatter(cluster[:, 1], cluster[:, 0], color='g', s=5)
        for defect in defects:
            i, j, defect_type = defect
            color = 'r' if defect_type > 0 else 'b'
            ax1.scatter(j, i, color=color)
    ax1.set_aspect('equal')

    return ax0, ax1

def theta_cal(d11, d12):
    return 0.5 * np.arctan2(2*d12, 2*d11-1)

def S_cal(d11, d12):
    q11 = d11 - 0.5
    q12 = d12
    S = 2 * np.sqrt(q11**2 + q12**2)
    return S


if __name__ == "__main__":
    PATH = '/home/hou63/pj1/code/test1/reference_data_D_20.0_zeta_2_seed918_.mat'
    data = sio.loadmat(PATH)
    D = data['D']
    D_11 = D[:,:,0,:]
    D_12 = D[:,:,1,:]
    D_22 = D[:,:,2,:]

    ind = 66

    d11 = D_11[:,:,ind]
    d12 = D_12[:,:,ind]
    d22 = D_22[:,:,ind]
    q11 = d11 - 0.5
    q12 = d12

    S = 2 * np.sqrt(q11**2 + q12**2)
    S_max = np.max(np.max(S))
    print(S_max)

    flag = S < 0.5*S_max

    theta = 0.5 * np.arctan2(2*d12, 2*d11-1)

    # defects_new = defects_analysis(theta, flag, grid_size=1, eps=3, min_pts=20)
    # plot flag



    