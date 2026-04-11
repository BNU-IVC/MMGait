import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel, get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from

from ..components import TCN_ST
from ..graph import Graph
import numpy as np

class GaitTR(BaseModel):

    def build_network(self, model_cfg):

        in_c = model_cfg['in_channels']
        self.num_class = model_cfg['num_class']
        self.joint_format = model_cfg['joint_format']
        self.graph = Graph(joint_format=self.joint_format,max_hop=3)

        #### Network Define ####

        # ajaceny matrix
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))

        #data normalization
        num_point = self.A.shape[-1]
        self.data_bn = nn.BatchNorm1d(in_c[0] * num_point)
        
        #backbone
        backbone = []
        for i in range(len(in_c)-1):
            backbone.append(TCN_ST(in_channel= in_c[i],out_channel= in_c[i+1],A=self.A,num_point=num_point))
        self.backbone = nn.ModuleList(backbone)

        self.fcn = nn.Conv1d(in_c[-1], self.num_class, kernel_size=1)
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

        ipts, labs, _, _,modal_valid, seqL = inputs

        x= ipts[0] 
        pose = x

        N, C, T, V, M = x.size()
        if len(x.size()) == 4:
            x = x.unsqueeze(1)
        del ipts

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        #backbone
        for _,m in enumerate(self.backbone):
            x = m(x)
        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1,V))
        #M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)#[n,c,t]
        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2]) #[n,c]
        # C fcn
        x = self.fcn(x) #[n,c']
        x = F.avg_pool1d(x, x.size()[2:]) # [n,c']
        x = x.view(N, self.num_class) # n,c
        embed = x.unsqueeze(-1) # n,c,1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs}
            },
            'visual_summary': {
                'image/pose': pose.view(N*T, M, V, C)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
