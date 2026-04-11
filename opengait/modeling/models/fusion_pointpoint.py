# 对于Multi和Cross在Dataset load数据的时候要if sum(modal_valid) == len(modal_valid): 
import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
import torch.nn.functional as F
from .lidargaitv2_utils import PointNetSetAbstraction, PPPooling, PPPooling_UDP,NetVLAD


from einops import rearrange
import copy
import cv2
from kornia import morphology as morph

blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class fusion_pointpoint(BaseModel):

    def build_network(self, model_cfg):

        ######### Points Encoder

        C = model_cfg['channel']
        C_out = 256
        scale_aware = model_cfg['scale_aware']
        normalize_dp = model_cfg['normalize_dp']
        sampling = model_cfg['sampling']

        npoints = model_cfg.get('npoints', [512, 256, 128])
        nsample = model_cfg.get('nsample', 32)
        in_channel = 4 if scale_aware else 3

        self.modal1_sa1 = PointNetSetAbstraction(npoint=npoints[0], radius=0.1, nsample=nsample, in_channel=in_channel, mlp=[2*C, 2*C, 4*C], group_all=False, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)
        self.modal1_sa2 = PointNetSetAbstraction(npoint=npoints[1], radius=0.2, nsample=nsample, in_channel=4*C + in_channel, mlp=[4*C, 4*C, 8*C], group_all=False, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)
        self.modal1_sa3 = PointNetSetAbstraction(npoint=npoints[2], radius=0.4, nsample=nsample, in_channel=8*C + in_channel, mlp=[8*C, 8*C, 16*C], group_all=False, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)
        self.modal1_sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=16*C + in_channel, mlp=[16*C, 16*C, C_out], group_all=True, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)

        self.modal2_sa1 = PointNetSetAbstraction(npoint=npoints[0], radius=0.1, nsample=nsample, in_channel=in_channel, mlp=[2*C, 2*C, 4*C], group_all=False, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)
        self.modal2_sa2 = PointNetSetAbstraction(npoint=npoints[1], radius=0.2, nsample=nsample, in_channel=4*C + in_channel, mlp=[4*C, 4*C, 8*C], group_all=False, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)
        self.modal2_sa3 = PointNetSetAbstraction(npoint=npoints[2], radius=0.4, nsample=nsample, in_channel=8*C + in_channel, mlp=[8*C, 8*C, 16*C], group_all=False, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)
        self.modal2_sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=16*C + in_channel, mlp=[16*C, 16*C, C_out], group_all=True, sampling=sampling, scale_aware=scale_aware, normalize_dp=normalize_dp)


        if model_cfg['pool'] == 'VLAD':
            self.modal1_pool = NetVLAD(num_clusters=16, dim=C_out, alpha=1.0)
            self.modal2_pool = NetVLAD(num_clusters=16, dim=C_out, alpha=1.0)
        elif model_cfg['pool'] == 'GMaxP':
            self.modal1_pool = PPPooling_UDP([1])
            self.modal2_pool = PPPooling_UDP([1])
        elif model_cfg['pool'] == 'PPP_UDP':
            self.modal1_pool = PPPooling_UDP(model_cfg['scale'])
            self.modal2_pool = PPPooling_UDP(model_cfg['scale'])
        elif model_cfg['pool'] == 'PPP_UAP':
            self.modal1_pool = PPPooling(scale_aware=False, bin_num=model_cfg['scale'])
            self.modal2_pool = PPPooling(scale_aware=False, bin_num=model_cfg['scale'])
        elif model_cfg['pool'] == 'PPP_HAP':
            self.modal1_pool = PPPooling(scale_aware=True, bin_num=model_cfg['scale'])
            self.modal2_pool = PPPooling(scale_aware=True, bin_num=model_cfg['scale'])
        



        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs']) 


    def point_forword(self, ipt, sa1, sa2, sa3, sa4, pool):
        B, T, N, C = ipt.shape
        xyz = rearrange(ipt, 'B T N C -> (B T) C N')
        l1_xyz, l1_points = sa1(xyz, None)
        l1_points = torch.max(l1_points, dim=-2)[0]

        l2_xyz, l2_points = sa2(l1_xyz, l1_points)
        l2_points = torch.max(l2_points, dim=-2)[0]

        l3_xyz, l3_points = sa3(l2_xyz, l2_points)
        l3_points = torch.max(l3_points, dim=-2)[0]

        l4_xyz, l4_points = sa4(l3_xyz, l3_points)

        x = pool(l4_points, l3_xyz)


        x = rearrange(x, '(B T) feat p -> B T feat p', B=B)
        feat = x.max(1)[0]# x.mean(1) # x.max(1)[0]

        return feat
    def forward(self, inputs):

        ipts, labs, typs, vies, modal_valid_, seqL = inputs
        modal_valid = list(map(list, zip(*modal_valid_)))
        modal_valid = torch.tensor(modal_valid, dtype=torch.bool, device=ipts[0].device)

        feat1 = self.point_forword(ipts[0], self.modal1_sa1, self.modal1_sa2, self.modal1_sa3, self.modal1_sa4, self.modal1_pool)
        feat2 = self.point_forword(ipts[1], self.modal2_sa1, self.modal2_sa2, self.modal2_sa3, self.modal2_sa4, self.modal2_pool)
        del ipts
        # print(feat1.shape)
        # print(feat2.shape)
        feat = torch.cat([feat1,feat2],dim=1)
        embed = self.FCs(feat)  # [n, c, p]
        _, logits = self.BNNecks(embed)  # [n, c, p]
 
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval


