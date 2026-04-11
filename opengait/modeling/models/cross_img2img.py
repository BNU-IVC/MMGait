import torch
import torch.nn as nn
import pdb
import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
import torch.nn.functional as F

from einops import rearrange
import copy
import cv2
from kornia import morphology as morph
blocks_map = {
    '2d': BasicBlock2D, 
    'p3d': BasicBlockP3D, 
    '3d': BasicBlock3D
}

class Cross_img2img(BaseModel):

    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        modal1_in_channels = model_cfg['Backbone']['modal1_in_channels']
        modal2_in_channels = model_cfg['Backbone']['modal2_in_channels']
        layers      = model_cfg['Backbone']['layers']
        channels    = model_cfg['Backbone']['channels']

        if mode == '3d': 
            strides = [
                [1, 1], 
                [1, 2, 2], 
                [1, 2, 2], 
                [1, 1, 1]
            ]
        else: 
            strides = [
                [1, 1], 
                [2, 2], 
                [2, 2], 
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.modal1_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(modal1_in_channels, self.inplanes, 1), 
            nn.BatchNorm2d(self.inplanes), 
            nn.ReLU(inplace=True)
        ))
        self.modal2_layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(modal2_in_channels, self.inplanes, 1), 
            nn.BatchNorm2d(self.inplanes), 
            nn.ReLU(inplace=True)
        ))
        self.modal1_layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))
        self.modal2_layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d': 
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.modal1_FCs = SeparateFCs(16, channels[3], channels[2])
        self.modal1_BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])
        self.modal2_FCs = SeparateFCs(16, channels[3], channels[2])
        self.modal2_BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def img_preteat(self,ipt):
        if len(ipt.size()) == 4:
            modal = ipt.unsqueeze(1)
        else:
            modal = ipt
            modal = modal.transpose(1, 2).contiguous()
        return modal
    def forward(self, inputs):

        ipts, labs, typs, vies, modal_valid_, seqL = inputs
        modal_valid = list(map(list, zip(*modal_valid_)))
        modal_valid = torch.tensor(modal_valid, dtype=torch.bool, device=ipts[0].device)

        modal1 = self.img_preteat(ipts[0])
        modal2 = self.img_preteat(ipts[1])

        del ipts


        out1 = self.modal1_layer0(modal1)
        out1 = self.modal1_layer1(out1)


        out2 = self.modal2_layer0(modal2)
        out2 = self.modal2_layer1(out2)

        if self.training:
            B = out1.shape[0]
            concat_feats = torch.cat([out1,out2],dim=0)
            concat_feats = self.layer2(concat_feats)
            concat_feats = self.layer3(concat_feats)
            concat_feats = self.layer4(concat_feats)

            out1, out2 = torch.split(concat_feats, B, dim=0)
        else:
            out1 = self.layer2(out1)
            out1 = self.layer3(out1)
            out1 = self.layer4(out1)

            out2 = self.layer2(out2)
            out2 = self.layer3(out2)
            out2 = self.layer4(out2)

        if seqL == None:
            # Temporal Pooling, TP
            out1 = self.TP(out1, seqL, options={"dim": 2})[0]  # [n, c, h, w]
            out2 = self.TP(out2, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        else:
            out1 = self.TP(out1, seqL[0].unsqueeze(0), options={"dim": 2})[0]  # [n, c, h, w]
            out2 = self.TP(out2, seqL[1].unsqueeze(0), options={"dim": 2})[0]  # [n, c, h, w]
        

        # Horizontal Pooling Matching, HPM
        feat1 = self.HPP(out1)  # [n, c, p]
        feat2 = self.HPP(out2)  # [n, c, p]

        embed1 = self.modal1_FCs(feat1)  # [n, c, p]
        _, logits1 = self.modal1_BNNecks(embed1)  # [n, c, p]

        embed2 = self.modal2_FCs(feat2)  # [n, c, p]
        _, logits2 = self.modal2_BNNecks(embed2)  # [n, c, p]


 
        retval = {
            'training_feat': {
                'cross_triplet1': {'embeddings1': embed1, 'embeddings2':embed2, 'labels':labs},
                'cross_triplet2': {'embeddings1': embed2, 'embeddings2':embed1, 'labels':labs},
                'softmax1': {'logits': logits1, 'labels': labs},
                'softmax2': {'logits': logits2, 'labels': labs}
            },
            'visual_summary': {
                # 'image/mo1': rearrange(modal1, 'n c s h w -> (n s) c h w'),
                # 'image/mo2': rearrange(modal2, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': torch.cat([embed1,embed2],dim = -1)
            }
        }
        

        return retval

