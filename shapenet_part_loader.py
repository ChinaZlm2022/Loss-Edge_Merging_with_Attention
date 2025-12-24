# from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import json
import numpy as np
import sys

from datasetCategory import pfnet_processing_dataset, pcn_processing_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.abspath(
    os.path.join(BASE_DIR, '../dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/'))


class PartDataset(data.Dataset):
    def __init__(self, root=dataset_path, which_dataset='', npoints=2500, classification=False,
                 class_choice=None, split='train',
                 normalize=True):
        self.npoints = npoints
        self.root = root
        if which_dataset == 'PCN':
            self.catfile = os.path.join(self.root, 'PCN.json')
        else:
            self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.classification = classification
        self.normalize = normalize
        with open(self.catfile, 'r') as f:
            if which_dataset == 'PCN':
                f = json.load(f)
            for line in f:
                if which_dataset == 'PCN':
                    if line['taxonomy_name'] == 'airplane':
                        self.cat[line['taxonomy_name']] = line['taxonomy_id']
                else:
                    ls = line.strip().split()
                    self.cat[ls[0]] = ls[1]
        if class_choice is not None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        self.meta = {}
        self.which_dataset = which_dataset
        if which_dataset == 'PCN':
            train_ids, val_ids, test_ids = pcn_processing_dataset(self)
        else:
            train_ids, val_ids, test_ids = pfnet_processing_dataset(self)

        for item in self.cat:
            self.meta[item] = []
            # self.cat[item]取出来的是物体类的代码，例如：026911156
            if which_dataset == 'PCN':
                if split == 'train':
                    dir_point = os.path.join(self.root, 'shapenet/train/complete', self.cat[item])
                    dir_seg = os.path.join(self.root, 'shapenet/train/complete', self.cat[item])
                elif split == 'test':
                    dir_point = os.path.join(self.root, 'shapenet/test/complete', self.cat[item])
                    dir_seg = os.path.join(self.root, 'shapenet/test/complete', self.cat[item])
            else:
                dir_point = os.path.join(self.root, self.cat[item], 'points')
                dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            fns = sorted(os.listdir(dir_point))
            # 此刻fns是/www/dataset/zlm/shapenet_part/shapenet_core._v0/026911156/
            # points/3d2cb9d291ec39dc58a42593b26221da.pts?排序的列表
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                sys.exit(-1)
            for fn in fns:
                # 1、os.path.basename(fn)是获取fn的文件名：例如，
                # fn是/www/dataset/zlm/shapenet_part/shapenet_core._v0/026911156/points
                # /3d2cb9d291ec39dc58a42593b26221da.pts
                # 那么该函数返回的是文件名3d2cb9d291ec39dc58a42593b26221da.pts
                # 2、os.path.splitext()是将3d2cb9d291ec39dc58a42593b26221da.pts分割成一个元组，
                # 为['3d2cb9d291ec39dc58a42593b26221da','.pts']
                token = (os.path.splitext(os.path.basename(fn))[0])
                # 这一步的作用是去除掉多余的fns后，再重新拼接有效的pts文件。
                if which_dataset == 'PCN':
                    self.meta[item].append(
                        (os.path.join(dir_point, token + '.pcd'), os.path.join(dir_seg, token + '.seg'),
                         self.cat[item], token))
                else:
                    self.meta[item].append(
                        (os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'),
                         self.cat[item], token))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2], fn[3]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath) // 50):
                l = len(np.unique(np.loadtxt(self.datapath[i][2]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 18000

    def __getitem__(self, index):
        if index in self.cache:
            #            point_set, seg, cls= self.cache[index]
            point_set, seg, cls, foldername, filename = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            #            cls = np.array([cls]).astype
            if self.which_dataset == 'PCN':
                point_set = np.loadtxt(fn[1], skiprows=11).astype(np.float32)

            else:
                point_set = np.loadtxt(fn[1]).astype(np.float32)

            if self.normalize:
                point_set = self.pc_normalize(point_set)

            if self.which_dataset == 'PCN':
                seg = np.loadtxt(fn[1], skiprows=11).astype(np.int64) - 1
            else:
                seg = np.loadtxt(fn[1]).astype(np.int64) - 1

            foldername = fn[3]
            filename = fn[4]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg, cls, foldername, filename)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        # To Pytorch
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:

            return point_set, cls
        else:
            return point_set, seg, cls

    def __len__(self):
        return len(self.datapath)

    def pc_normalize(self, pc):
        """ pc: NxC, return NxC """
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc


if __name__ == '__main__':
    dset = PartDataset(root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/', classification=True,
                       class_choice=None, npoints=4096, split='train')
    #    d = PartDataset( root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=False, class_choice=None, npoints=4096, split='test')
    print(len(dset))
    ps, cls = dset[10000]
    print(cls)
#    print(ps.size(), ps.type(), cls.size(), cls.type())
#    print(ps)
#    ps = ps.numpy()
#    np.savetxt('ps'+'.txt', ps, fmt = "%f %f %f")
