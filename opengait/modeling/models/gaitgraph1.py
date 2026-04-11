import torch
from ..base_model import BaseModel, get_valid_args, is_list, is_dict, np2var, ts2np, list2var, get_attr_from

from ..backbones.resgcn import ResGCN
from ..graph.graph import Graph
import torch.nn.functional as F
import numpy as np

class GaitGraph1(BaseModel):

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
        x_input = ipts[0] # N C T V I
        x_input = x_input.permute(0, 2, 3, 4, 1).contiguous()
        N, T, V, I, C = x_input.size() 
        
        pose  = x_input
        if self.training:
            x_input = torch.cat([x_input[:,:int(T/2),...],x_input[:,int(T/2):,...]],dim=0) #[8, 60, 17, 1, 3]
        elif self.tta:
            data_flipped = torch.flip(x_input,dims=[1])
            x_input = torch.cat([x_input,data_flipped], dim=0)

        x = x_input.permute(0, 3, 4, 1, 2).contiguous()

        # resgcn
        x = self.ResGCN(x)
        x = F.normalize(x, dim=1, p=2) # norm #only for GaitGraph1 # Remove from GaitGraph2
        
        if self.training:
            f1, f2 = torch.split(x, [N, N], dim=0)
            embed = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) #[4, 2, 128]
            
        elif self.tta:
            f1, f2 = torch.split(x, [N, N], dim=0)
            embed = torch.mean(torch.stack([f1, f2]), dim=0)
            embed = embed.unsqueeze(-1)
        else:
            embed = embed.unsqueeze(-1)
        
        retval = {
            'training_feat': {
                'SupConLoss': {'features': embed , 'labels': labs}, # loss
            },
            'visual_summary': {
                'image/pose': pose.view(N*T, 1, I*V, C).contiguous() # visualization
            },
            'inference_feat': {
                'embeddings':   embed # for metric
            }
        }
        return retval
