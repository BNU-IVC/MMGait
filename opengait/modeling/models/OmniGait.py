import torch
import torch.nn as nn
import numpy as np


from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D



from einops import rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModalFusion2D(nn.Module):

    def __init__(self, in_channels=128, num_modal=2):
        super().__init__()
        self.num_modal = num_modal
        self.in_channels = in_channels
        self.fusion_conv = nn.Conv2d(in_channels * num_modal, in_channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_modal, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, concat_feat):

        feats_list = torch.split(concat_feat,self.in_channels,dim=1)
        fused = self.fusion_conv(concat_feat)        # [B, C, H, W]

        gate_weights = self.gate(fused)        # [B, num_modal, 1, 1]
        gate_weights = torch.softmax(gate_weights, dim=1)

        weighted_sum = 0
        for i in range(self.num_modal):
            weighted_sum += feats_list[i] * gate_weights[:, i:i+1]

        out = fused + weighted_sum
        return out

class OmniGait(BaseModel):
    fusion_pairs = [
        ("rgb_sils", "depth"),
        ("rgb_sils", "event"),
        ("rgb_sils", "heatmap"),
        ("rgb_sils", "ir"),
        ("rgb_sils", "ir_sils"),
        ("rgb_sils", "lidar_depth"),
        ("rgb_sils", "radar_depth"),
        ("rgb_sils", "rgb"),
    ]

    def img_pretreat(self,ipt):
        if len(ipt.size()) == 4:
            modal = ipt.unsqueeze(1)
        else:
            modal = ipt
            modal = modal.transpose(1, 2).contiguous()
        return modal

    def make_layer(self, block, inplanes, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
        else:
            downsample = lambda x: x
        layers = [block(inplanes, planes, stride=stride, downsample=downsample)]
        return nn.Sequential(*layers)

    def bulid_img_encoder(self, model_cfg):
        # Input 
        block = BasicBlock2D
        layers      = [1,1,1]
        channels    = [128,256,512]
        strides = [
            [2, 2], 
            [2, 2], 
            [1, 1]
        ]
        layer2 = self.make_layer(block, 128, channels[0], strides[0], blocks_num=layers[0])
        layer3 = self.make_layer(block, channels[0], channels[1], strides[1], blocks_num=layers[1])
        layer4 = self.make_layer(block, channels[1], channels[2], strides[2], blocks_num=layers[2])
        layer2 = SetBlockWrapper(layer2)
        layer3 = SetBlockWrapper(layer3)
        layer4 = SetBlockWrapper(layer4)
        self.img_encoder = nn.Sequential(
            layer2,
            layer3,
            layer4
        )
    
        
    def make_tokenizer(self, in_channels):
        layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, 64, 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
        ))
        layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, 64, 128, [1,1], blocks_num=1, mode='2d'))

        return nn.Sequential(
            layer0,
            layer1
        )

    def bulid_tokenizer(self):
        self.depth_tokenizer = self.make_tokenizer(3)
        self.event_tokenizer = self.make_tokenizer(3)
        self.heatmap_tokenizer = self.make_tokenizer(2)
        self.ir_tokenizer = self.make_tokenizer(3)
        self.ir_sils_tokenizer = self.make_tokenizer(1)
        self.lidar_depth_tokenizer = self.make_tokenizer(3)
        self.radar_depth_tokenizer = self.make_tokenizer(3)
        self.rgb_tokenizer = self.make_tokenizer(3)
        self.rgb_sils_tokenizer = self.make_tokenizer(1)



    def build_network(self, model_cfg):

        self.bulid_img_encoder(model_cfg['img_encoder'])
        self.bulid_tokenizer()


        self.FCs = SeparateFCs(16, 512, 256)
        self.BNNecks = SeparateBNNecks(16, 256, class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

        self.early_fusion_module = SetBlockWrapper(ConvModalFusion2D(in_channels=128, num_modal=2))


    def concat_forword(self, feat, seqL):
        feat = self.img_encoder(feat)# [B, c, s, h, w]
        feat = self.TP(feat, seqL, options={"dim": 2})[0]  # [B, c, h, w]
        feat = self.HPP(feat) # [B, c, p]
        embed = self.FCs(feat)  # [B, c, p]
        _, logits = self.BNNecks(embed)  # [B, c, p]
        return embed, logits

    def encode_token_feat(self, feat, seqL):
        feat = self.img_encoder(feat)
        feat = self.TP(feat, seqL, options={"dim": 2})[0]
        feat = self.HPP(feat)
        embed = self.FCs(feat)
        _, logits = self.BNNecks(embed)
        return embed, logits

    def split_forword(self, img, tokenizer, seqL):
        feat = tokenizer(img)# [B, c, s, h, w]
        return self.encode_token_feat(feat, seqL)

    def pad_token_feat(self, feat, target_len):
        curr_len = feat.size(2)
        if curr_len >= target_len:
            return feat
        if curr_len == 0:
            pad_shape = list(feat.shape)
            pad_shape[2] = target_len
            return feat.new_zeros(pad_shape)
        repeat_times = (target_len + curr_len - 1) // curr_len
        repeat_dims = [1] * feat.dim()
        repeat_dims[2] = repeat_times
        return feat.repeat(*repeat_dims).narrow(2, 0, target_len)

    def align_token_feats_for_fusion(self, anchor_feat, pair_feat, anchor_seqL, pair_seqL):
        if anchor_seqL is None or pair_seqL is None:
            target_len = max(anchor_feat.size(2), pair_feat.size(2))
            return (
                self.pad_token_feat(anchor_feat, target_len),
                self.pad_token_feat(pair_feat, target_len),
                None
            )

        anchor_lengths = anchor_seqL.reshape(-1).detach().cpu().tolist()
        pair_lengths = pair_seqL.reshape(-1).detach().cpu().tolist()
        fused_seqL = torch.maximum(anchor_seqL, pair_seqL)

        if anchor_feat.size(0) != 1 or pair_feat.size(0) != 1 or len(anchor_lengths) != len(pair_lengths):
            target_len = int(fused_seqL.max().item())
            return (
                self.pad_token_feat(anchor_feat, target_len),
                self.pad_token_feat(pair_feat, target_len),
                fused_seqL
            )

        anchor_starts = [0] + np.cumsum(anchor_lengths).tolist()[:-1]
        pair_starts = [0] + np.cumsum(pair_lengths).tolist()[:-1]
        fused_lengths = fused_seqL.reshape(-1).detach().cpu().tolist()

        aligned_anchor_feats = []
        aligned_pair_feats = []
        for anchor_start, pair_start, anchor_len, pair_len, fused_len in zip(
            anchor_starts, pair_starts, anchor_lengths, pair_lengths, fused_lengths
        ):
            anchor_slice = anchor_feat.narrow(2, int(anchor_start), int(anchor_len))
            pair_slice = pair_feat.narrow(2, int(pair_start), int(pair_len))
            aligned_anchor_feats.append(self.pad_token_feat(anchor_slice, int(fused_len)))
            aligned_pair_feats.append(self.pad_token_feat(pair_slice, int(fused_len)))

        return (
            torch.cat(aligned_anchor_feats, dim=2),
            torch.cat(aligned_pair_feats, dim=2),
            fused_seqL
        )

        

    def forward(self, inputs):

        ipts, labs, typs, vies, modal_valid_, seqL = inputs
        modal_valid = list(map(list, zip(*modal_valid_)))
        modal_valid = torch.tensor(modal_valid, dtype=torch.bool, device=ipts[0].device)
        
        # pretreat
        depth = self.img_pretreat(ipts[0]) # [B, C, S, H, W]
        event = self.img_pretreat(ipts[1])
        heatmap = self.img_pretreat(ipts[2])
        ir = self.img_pretreat(ipts[3])
        ir_sils = self.img_pretreat(ipts[4])
        lidar_depth = self.img_pretreat(ipts[5])
        radar_depth = self.img_pretreat(ipts[6])
        rgb = self.img_pretreat(ipts[7])
        rgb_sils = self.img_pretreat(ipts[8])
        del ipts

        # tokenize
        img_modals = [
            ("depth", depth, self.depth_tokenizer),
            ("event", event, self.event_tokenizer),
            ("heatmap", heatmap, self.heatmap_tokenizer),
            ("ir", ir, self.ir_tokenizer),
            ("ir_sils", ir_sils, self.ir_sils_tokenizer),
            ("lidar_depth", lidar_depth, self.lidar_depth_tokenizer),
            ("radar_depth", radar_depth, self.radar_depth_tokenizer),
            ("rgb", rgb, self.rgb_tokenizer),
            ("rgb_sils", rgb_sils, self.rgb_sils_tokenizer)
        ]


        if self.training:
            token_feats= []
            for _, modal_input, tokenizer in img_modals:
                tf = tokenizer(modal_input)   # shape: [B, C, S, H, W]
                token_feats.append(tf) 

            fused_feats = [self.early_fusion_module(torch.cat([token_feats[-1],token_feats[j]],dim=1)) for j in range(len(token_feats)-1)] #8 * Fused Feature

            all_feats = token_feats + fused_feats
            concat_feats = torch.cat(all_feats, dim=0)  # (B * num_img_modals, C, S, H, W)
            all_embeds, all_logits = self.concat_forword(concat_feats,seqL)  # (B * num_img_modals, C, P)

            all_labs = torch.cat([labs,labs,labs,labs,labs,labs,labs,labs,labs,labs,labs,labs,labs,labs,labs,labs,labs],dim=0)

        else:

            # Omni-eval
            token_feat_dict = {}
            seqL_dict = {}
            all_embeds = []
            for (i, (modal_name, modal_input, tokenizer)) in enumerate(img_modals):
                token_feat_dict[modal_name] = tokenizer(modal_input)
                seqL_dict[modal_name] = seqL[i].unsqueeze(0)
                embed, logits = self.split_forword(modal_input, tokenizer, seqL[i].unsqueeze(0))   # shape: [B, C, ...]
                all_embeds.append(embed)

            fusion_embeds = []
            for anchor_name, pair_name in self.fusion_pairs:
                anchor_feat, pair_feat, fused_seqL = self.align_token_feats_for_fusion(
                    token_feat_dict[anchor_name],
                    token_feat_dict[pair_name],
                    seqL_dict[anchor_name],
                    seqL_dict[pair_name]
                )
                fused_feat = self.early_fusion_module(
                    torch.cat([anchor_feat, pair_feat], dim=1)
                )
                fusion_embed, _ = self.encode_token_feat(fused_feat, fused_seqL)
                fusion_embeds.append(fusion_embed)

            all_embeds = torch.cat(all_embeds,dim=-1)
            fusion_embeds = torch.cat(fusion_embeds, dim=-1)
            all_labs = None
            all_logits = None


        retval = {
            'training_feat': {
                'softmax': {'logits': all_logits, 'labels': all_labs},
                'triplet': {'embeddings': all_embeds,  'labels': all_labs }
            },
            'visual_summary': {

            },
            'inference_feat': {
                'embeddings': all_embeds,
                'fusion_embeddings': fusion_embeds if not self.training else all_embeds
            }
        }

        return retval
