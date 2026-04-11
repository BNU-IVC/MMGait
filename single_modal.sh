### 2D Pose ### 
# GPGait #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/pose2d/GPGait.yaml --phase train --log_to_file
# GPGait++ #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/pose2d/GPGait++.yaml --phase train --log_to_file
# GaitTR #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/pose2d/GaitTR.yaml --phase train --log_to_file
# GaitGraph1 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/pose2d/GaitGraph1.yaml --phase train --log_to_file
# GaitGraph2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/pose2d/GaitGraph2.yaml --phase train --log_to_file
# SkeletonGait #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/pose2d/SkeletonGait.yaml --phase train --log_to_file



### RGB_Sils ###
# DeepGaitV2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/rgb_sils/DeepGaitV2.yaml --phase train --log_to_file
# GaitBase #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/rgb_sils/GaitBase.yaml --phase train --log_to_file
# GaitGL #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/rgb_sils/GaitGL.yaml --phase train --log_to_file
# GaitPart #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/rgb_sils/GaitPart.yaml --phase train --log_to_file
# GaitSet #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/rgb_sils/GaitSet.yaml --phase train --log_to_file






# ### RGB ###
# # DeepGaitV2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/rgb/DeepGaitV2.yaml --phase train --log_to_file

# ### IR ###
# # DeepGaitV2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/ir/DeepGaitV2.yaml --phase train --log_to_file

# ### Depth ###
# # DeepGaitV2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/depth/DeepGaitV2.yaml --phase train --log_to_file

# ### Lidar Depth ### 
# # DeepGaitV2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/lidar_depth/DeepGaitV2.yaml --phase train --log_to_file

# ## Radar Depth ### 
# # DeepGaitV2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/radar_depth/DeepGaitV2.yaml --phase train --log_to_file

### Event ###
# DeepGaitV2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/event/DeepGaitV2.yaml --phase train --log_to_file

### IR_Sils ###
# DeepGaitV2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/ir_sils/DeepGaitV2.yaml --phase train --log_to_file


### 3D Pose ### 
# GPGait #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/pose3d/GPGait.yaml --phase train --log_to_file
# GPGait++ #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/pose3d/GPGait++.yaml --phase train --log_to_file

### Lidar Points ###
# LidarGaitv2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/lidar_points/lidargaitv2.yaml --phase train --log_to_file


### Radar Points ###
# LidarGaitv2 #
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py --cfgs ./configs/singlemodal/radar_points/lidargaitv2.yaml --phase train --log_to_file


