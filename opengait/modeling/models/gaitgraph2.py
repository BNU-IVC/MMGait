import torch
import torch.nn as nn
from ..base_model import BaseModel, get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from

from ..backbones.resgcn import ResGCN
from ..graph.graph import Graph
import numpy as np


import numpy as np


class GaitGraph2(BaseModel):

    def build_network(self, model_cfg):
         
        self.joint_format = model_cfg['joint_format']
        self.input_num = model_cfg['input_num']
        self.block = model_cfg['block']
        self.input_branch = model_cfg['input_branch']
        self.main_stream = model_cfg['main_stream']
        self.num_class = model_cfg['num_class']
        self.reduction = model_cfg['reduction']
        self.tta = model_cfg['tta']
        ## Graph Init ##
        self.graph = Graph(joint_format=self.joint_format,max_hop=3)
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        ## Network ##
        self.ResGCN = ResGCN(input_num=self.input_num, input_branch=self.input_branch, 
                             main_stream=self.main_stream, num_class=self.num_class,
                             reduction=self.reduction, block=self.block,graph=self.A)

    def inputs_pretreament(self, inputs):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, modal_valid_batch,  seqL_batch = inputs
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        requires_grad = bool(self.training)
        seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch
        vies = vies_batch


        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:,:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, modal_valid_batch, seqL

    def forward(self, inputs):

        ipts, labs, type_, view_, modal_valid, seqL = inputs
        x_input = ipts[0] 
        N, T, V, I, C = x_input.size()
        pose  = x_input
        flip_idx = self.graph.flip_idx

        if not self.training and self.tta:
            multi_input = MultiInput(self.graph.connect_joint, self.graph.center)
            x1 = []
            x2 = []
            for i in range(N):
                x1.append(multi_input(x_input[i,:,:,0,:3].flip(0)))
                x2.append(multi_input(x_input[i,:,flip_idx,0,:3]))
            x_input = torch.cat([x_input, torch.stack(x1,0), torch.stack(x2,0)], dim=0)
        
        x = x_input.permute(0, 3, 4, 1, 2).contiguous()

        # resgcn
        x = self.ResGCN(x)

        if not self.training and self.tta:
            f1, f2, f3 = torch.split(x, [N, N, N], dim=0)
            x = torch.cat((f1, f2, f3), dim=1)
             
        embed = torch.unsqueeze(x,-1)
        
        retval = {
            'training_feat': {
                'SupConLoss': {'features': x , 'labels': labs}, # loss
            },
            'visual_summary': {
                'image/pose': pose.view(N*T, 1, I*V, C).contiguous() # visualization
            },
            'inference_feat': {
                'embeddings': embed # for metric
            }
        }
        return retval
    
class MultiInput:
    def __init__(self, connect_joint, center):
        self.connect_joint = connect_joint
        self.center = center

    def __call__(self, data):

        # T, V, C -> T, V, I=3, C + 2
        T, V, C = data.shape
        x_new = torch.zeros((T, V, 3, C + 2), device=data.device)

        # Joints
        x = data
        x_new[:, :, 0, :C] = x
        for i in range(V):
            x_new[:, i, 0, C:] = x[:, i, :2] - x[:, self.center, :2]

        # Velocity
        for i in range(T - 2):
            x_new[i, :, 1, :2] = x[i + 1, :, :2] - x[i, :, :2]
            x_new[i, :, 1, 3:] = x[i + 2, :, :2] - x[i, :, :2]
        x_new[:, :, 1, 3] = x[:, :, 2]

        # Bones
        for i in range(V):
            x_new[:, i, 2, :2] = x[:, i, :2] - x[:, self.connect_joint[i], :2]
        bone_length = 0
        for i in range(C - 1):
            bone_length += torch.pow(x_new[:, :, 2, i], 2)
        bone_length = torch.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            x_new[:, :, 2, C+i] = torch.acos(x_new[:, :, 2, i] / bone_length)
        x_new[:, :, 2, 3] = x[:, :, 2]

        data = x_new
        return data

