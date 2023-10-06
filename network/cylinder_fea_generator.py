# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import gen_voxel_label
import gen_pt_ind
import time


class cylinder_fea(nn.Module):

    def __init__(self, grid_size, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, pt_lab, xyz, batch_size, configs):
        cur_dev = pt_fea[0].get_device()

        # concate everything
        cat_xyz = []
        for i_batch in range(len(xyz)):
            cat_xyz.append(F.pad(xyz[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_lab = torch.cat(pt_lab, dim=0)
        cat_xyz = torch.cat(cat_xyz, dim=0)
        pt_num = cat_xyz.shape[0]
        
        # gen grid index and augment feature (concat relative coordinate)
        max_bound = torch.tensor(configs['dataset_params']['max_volume_space'], device=cur_dev)
        min_bound = torch.tensor(configs['dataset_params']['min_volume_space'], device=cur_dev)
        
        crop_range = max_bound - min_bound
        grid_size = torch.tensor(configs['model_params']['output_shape'], device=cur_dev)
        intervals = crop_range / (grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")
        
        cat_pt_ind = torch.zeros_like(cat_xyz, device=cur_dev)
        cat_xyz_clone = cat_xyz.clone()
        cat_xyz_clone[:, 1:] = cat_xyz_clone[:, 1:].clamp(min_bound, max_bound)
        gen_pt_ind.gen_pt_ind_cuda(cat_pt_ind, cat_xyz_clone, min_bound, intervals)
        cat_pt_ind = cat_pt_ind.floor().to(dtype=torch.int64)
        voxel_centers = (cat_pt_ind[:, 1:].to(dtype=torch.float32) + 0.5) * intervals + min_bound
        rel_xyz = cat_xyz[:, 1:] - voxel_centers
        cat_pt_fea = torch.concat((rel_xyz, cat_pt_fea), dim=1)
        
        # gen voxel label
        voxel_labels = torch.ones((batch_size, *grid_size, configs['model_params']['num_class']), 
                                  dtype=torch.int32, device=cur_dev) * configs['dataset_params']['ignore_label']
        gen_voxel_label.gen_voxel_label_cuda(voxel_labels, cat_pt_lab, cat_pt_ind)
        voxel_labels = voxel_labels.argmax(dim=-1)   # argmax取完就是int64
        
        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        # cat_pt_ind = cat_pt_ind[shuffled_ind, :]
    
        # unique xy grid index
        unq, unq_inv = torch.unique(cat_pt_ind, return_inverse=True, return_counts=False, dim=0)
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]   # 只是非空voxel的feature

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        return unq, processed_pooled_data, voxel_labels, cat_pt_ind
