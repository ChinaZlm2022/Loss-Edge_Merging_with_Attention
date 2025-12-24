import open3d as o3d
import numpy as np
# import matplotlib.pyplot as plt
import codecs
from sklearn import datasets
import pandas as pd
from sklearn.cluster import dbscan
import time
import os
from os import path

# filecp = codecs.open('../Pointnet_Pointnet2_pytorch-master-原版/1/1.txt', encoding ='utf-8')
# file_data = np.loadtxt(filecp, usecols=(0,1,2),skiprows=1,delimiter=" ")
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(file_data)
# url = 'data2'
url = "/www/dataset/zlm/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/"


# url ='data1'

# 定义一个函数
def scaner_file(url):
    # 遍历当前路径下所有文件
    file = os.listdir(url)
    for f in file:
        # real_url是points的路径
        real_url = path.join(url, f + '/points')
        per_data_file = os.listdir(real_url)
        for per_pts in per_data_file:
            per_pts_file = real_url + '/' + per_pts
            with open(per_pts_file) as per_pts__file_obj:
                contents = per_pts__file_obj.readlines()
            for per_file_line in contents:
                per_file_line = per_file_line.strip("\n").split(" ")
                data = list(map(float, per_file_line))
                pcd = o3d.geometry.PointCloud()
                print('数组数据111============', [data])
                pcd.points = o3d.utility.Vector3dVector([data])
                xyz2 = np.asarray(pcd.points)
                print('22222222===============xyz2', xyz2)

        # filecp = codecs.open(real_url, encoding='utf-8')

        # file_data = np.loadtxt(real_url, usecols=(0, 1, 2, 3, 4, 5, 6), skiprows=1, delimiter=" ")
        # file_data1 = file_data[:, :3]

        # keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
        # pcd = np.array(keypoints.points)
        # print(type(pcd))
        # m = pcd.shape[0]
        # n = file_data.shape[0]
        # c = []
        # for i in range(m):
        #     for j in range(n):
        #         if pcd[i, 0] == file_data[j, 0]:
        #             if pcd[i, 1] == file_data[j, 1]:
        #                 if pcd[i, 2] == file_data[j, 2]:
        #                     c.append(file_data[j, 6])
        #                     break
        #                 else:
        #                     continue
        #             else:
        #                 continue
        #         else:
        #             continue
        # c = np.array(c)
        # c = c[:, np.newaxis]
        # print(c.shape)
        # print(pcd.shape)
        # pcd = np.append(pcd, c, axis=1)
        # j = path.join(url, "01" + f)
        # np.savetxt(j, pcd, fmt="%f")

        # print(real_url)


# 这个函数对算法没什么影响，只是突出表现一下特征点

# def keypoints_to_spheres(keypoints):
#         spheres = o3d.geometry.TriangleMesh()
#         for keypoint in keypoints.points:
#          sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
#          sphere.translate(keypoint)
#          spheres += sphere
#          spheres.paint_uniform_color([1.0, 0.0, 0.0])
#         return spheres

#
# tic = time.time()
# # --------------------ISS关键点提取的相关参数---------------------
# keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
#                                                         # salient_radius=0.01,
#                                                         # non_max_radius=0.01,
#                                                         # gamma_21=0.8,
#                                                      # gamma_32=0.5)
# toc = 1000 * (time.time() - tic)
# print("ISS Computation took {:.0f} [ms]".format(toc))
# print("Extract",keypoints)
# pcd.paint_uniform_color([0.0, 1.0, 0.0])
# o3d.visualization.draw_geometries([keypoints_to_spheres(keypoints), pcd],width=800,height=800)

x = scaner_file(url)
