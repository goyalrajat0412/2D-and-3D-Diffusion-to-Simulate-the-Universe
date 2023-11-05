import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import open3d as o3d
import numpy as np
import torch.nn.functional as F

class Uniform15KPC(Dataset):
    def __init__(self, data="/home/ppydj3/Quijote/quijote15k.npy", tr_sample_size=10000, #/content/drive/MyDrive/Data/quijote15k.npy for colab
                 te_sample_size=10000, split='train', scale=1.,
                 normalize_per_shape=False, box_per_shape=False,
                 random_subsample=False,
                 normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None,
                 input_dim=3):
        self.data = np.load(data)
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.scale = scale
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        self.box_per_shape = box_per_shape
        #self.cate_idx_lst = []
        self.all_points = []
        for i in range(len(self.data)):
          point_cloud = self.data[i]
          assert point_cloud.shape[0] == 15000
          self.all_points.append(point_cloud[np.newaxis, ...])
          
        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        #self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        if all_points_mean is not None and all_points_std is not None:  # using loaded dataset stats
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:  # per shape normalization
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        elif self.box_per_shape:
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.min(axis=1).reshape(B, 1, input_dim)

            self.all_points_std = self.all_points.max(axis=1).reshape(B, 1, input_dim) - self.all_points.min(axis=1).reshape(B, 1, input_dim)

        else:  # normalize across the dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        if self.box_per_shape:
            self.all_points = self.all_points - 0.5
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def get_pc_stats(self, idx):
        if self.normalize_per_shape or self.box_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s


        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()

        m, s = self.get_pc_stats(idx)
        #cate_idx = self.cate_idx_lst[idx]
        #sid, mid = self.all_cate_mids[idx]

        out = {
            'idx': idx,
            'train_points': tr_out,
            'test_points': te_out,
            'mean': m, 'std': s
        }

        return out


class Quijote15kPointClouds(Uniform15KPC):
    def __init__(self, data="/home/ppydj3/Quijote/quijote15k.npy", #check above for colab
                 tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 normalize_std_per_axis=False, box_per_shape=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):
        self.data = np.load(data)
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]

        super(Quijote15kPointClouds, self).__init__(
            data,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split, scale=scale,
            normalize_per_shape=normalize_per_shape, box_per_shape=box_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean, all_points_std=all_points_std,
            input_dim=3)



class PointCloudMasks(object):
    '''
    render a view then save mask
    '''
    def __init__(self, radius : float=10, elev: float =45, azim:float=315, ):

        self.radius = radius
        self.elev = elev
        self.azim = azim


    def __call__(self, points):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        camera = [self.radius * np.sin(90-self.elev) * np.cos(self.azim),
                  self.radius * np.cos(90 - self.elev),
                  self.radius * np.sin(90 - self.elev) * np.sin(self.azim),
                  ]
        # camera = [0,self.radius,0]
        _, pt_map = pcd.hidden_point_removal(camera, self.radius)

        mask = torch.zeros_like(points)
        mask[pt_map] = 1

        return mask #points[pt_map]


####################################################################################

