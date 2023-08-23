# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
import spconv
import torch
from torch import nn
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, indice_key=None):
        super(ResContextBlock, self).__init__()
        
        self.conv1 = spconv.SparseSequential(conv1x3(in_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.LeakyReLU(),
                                             conv3x1(out_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.LeakyReLU())
        self.conv2 = spconv.SparseSequential(conv3x1(in_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.LeakyReLU(),
                                             conv1x3(out_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.LeakyReLU())

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        resA = self.conv2(x)
        resA = Fsp.sparse_add(resA, shortcut)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        
        self.conv1 = spconv.SparseSequential(conv3x1(in_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.LeakyReLU(),
                                             conv1x3(out_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.LeakyReLU())
        
        self.conv2 = spconv.SparseSequential(conv1x3(in_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.LeakyReLU(),
                                             conv3x1(out_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.LeakyReLU())

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        resA = self.conv2(x)
        resA = Fsp.sparse_add(resA, shortcut)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        
        self.trans = spconv.SparseSequential(conv3x3(in_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.LeakyReLU())
        
        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)
        
        self.conv = spconv.SparseSequential(conv1x3(out_filters, out_filters),
                                            nn.BatchNorm1d(out_filters),
                                            nn.LeakyReLU(),
                                            conv3x1(out_filters, out_filters),
                                            nn.BatchNorm1d(out_filters),
                                            nn.LeakyReLU(),
                                            conv3x3(out_filters, out_filters),
                                            nn.BatchNorm1d(out_filters),
                                            nn.LeakyReLU())
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans(x)

        # upsample
        upA = self.up_subm(upA)
        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv(upA)
        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()

        self.conv1 = spconv.SparseSequential(conv3x1x1(in_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.Sigmoid())
        
        self.conv2 = spconv.SparseSequential(conv1x3x1(in_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.Sigmoid())
        
        self.conv3 = spconv.SparseSequential(conv1x1x3(in_filters, out_filters),
                                             nn.BatchNorm1d(out_filters),
                                             nn.Sigmoid())

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut2 = self.conv2(x)
        shortcut3 = self.conv3(x)
        shortcut = Fsp.sparse_add(shortcut, shortcut2, shortcut3)
        
        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), dim=1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y
