CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/omnimodal/OmniGait.yaml --phase train --log_to_file
