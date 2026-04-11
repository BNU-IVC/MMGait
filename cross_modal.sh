# RGB (Sils)
# IR (Sils)
# Depth (Depth map)
# Lidar (Lidar Projected Depth)
# Radar (Radar Projected Depth)

# ## RGB-to-Depth ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/crossmodal/RGB-to-Depth/Sils-to-Depthmap/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file

# ### RGB-to-IR ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/crossmodal/RGB-to-IR/Sils-to-IR/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file

# ### RGB-to-Lidar ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7237 opengait/main.py --cfgs ./configs/crossmodal/RGB-to-Lidar/Sils-to-ProjetedDepth/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file

# ### IR-to-Depth ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/crossmodal/IR-to-Depth/IR-to-Depthmap/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file

# ### IR-to-Lidar ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/crossmodal/IR-to-Lidar/IR-to-ProjetedDepth/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file

# ### Depth-to-Lidar ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/crossmodal/Depth-to-Lidar/Depth-to-ProjetedDepth/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file

### RGB-to-Radar ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7237 opengait/main.py --cfgs ./configs/crossmodal/RGB-to-Radar/Sils-to-ProjectedDepth/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file

### IR-to-Radar ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7237 opengait/main.py --cfgs ./configs/crossmodal/IR-to-Radar/Sils-to-ProjectedDepth/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file

### Depth-to-Radar ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7237 opengait/main.py --cfgs ./configs/crossmodal/Depth-to-Radar/Depths-to-ProjectedDepth/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file

### LiDAR-to-Radar ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7237 opengait/main.py --cfgs ./configs/crossmodal/Lidar-to-Radar/ProjectedDepths-to-ProjectedDepth/DeepGaitV2_DeepGaitV2.yaml --phase train --log_to_file
