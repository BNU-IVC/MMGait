
# Intra-sensor
# RGB Camera
### Sils-and-Pose(Heatmap) ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7307 opengait/main.py --cfgs ./configs/multimodal/intra-sensor/Sils-and-Pose/MultiGait++.yaml --phase train --log_to_file

### Sils-and-Event ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7307 opengait/main.py --cfgs ./configs/multimodal/intra-sensor/Sils-and-Event/MultiGait++.yaml --phase train --log_to_file


# Inter-sensor
### Sils(RGB)-and-Depthmap ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7307 opengait/main.py --cfgs ./configs/multimodal/inter-sensor/RGB-and-Depth/MultiGait++.yaml --phase train --log_to_file

### Sils(RGB)-and-Projected Depth ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7307 opengait/main.py --cfgs ./configs/multimodal/inter-sensor/RGB-and-Lidar/MultiGait++.yaml --phase train --log_to_file

### Lidar Points-and-Radar Points ###
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7307 opengait/main.py --cfgs ./configs/multimodal/inter-sensor/Lidar-and-Radar/TwoStream.yaml --phase train --log_to_file
