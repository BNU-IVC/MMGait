# MMGait

**MMGait** is a multi-modal benchmark and codebase for **single-modal**, **cross-modal**, **multi-modal**, and **omni multi-modal** gait recognition.
<div align="center"><img src="assets/overview.png" alt="overview" width="900" /></div>




## Getting Started

### For the basic usage of MMGait
```bash
git clone https://github.com/BNU-IVC/MMGait
cd MMGait
```

#### 1. Installation
Recommended: Linux + CUDA + multi-GPU.

```bash
conda create -n mmgait python=3.10 -y
conda activate mmgait
pip install -r requirements.txt
```


#### 2. Download MMGait (Agreement Required)
To obtain and use MMGait and its subsets, all users are required to complete the following steps:

1. Download the latest agreement: `MMGait Dataset Usage Agreement.pdf`.
2. Complete and sign it.
3. Submit it to `BNU-IVC@outlook.com`.
4. Please use your organization/institute email address to send the mail.
5. We will handle requests within a week. Occasionally, emails may be flagged as spam. If you have not received a response within a week, please resend your mail from an alternate email address.

In case you encounter any issues, please feel free to reach out to us via `BNU-IVC@outlook.com`.

#### 3. Data Preparation

The loader expects the following directory layout:

```text
MMGait_ROOT/
`-- {subject_id}/
    `-- {seq_type}/
        `-- {view_id}/
            |-- depth.pkl
            |-- event.pkl
            |-- heatmap.pkl
            |-- ir.pkl
            |-- ir_sils.pkl
            |-- lidar_depth.pkl
            |-- lidar_points.pkl
            |-- pose2d.pkl
            |-- pose3d.pkl
            |-- radar4d.pkl
            |-- radar_depth.pkl
            |-- rgb.pkl
            `-- rgb_sils.pkl
```

Set it to your local dataset path in the config you run.

`data_in_use` follows this fixed order:

```text
[depth, event, heatmap, ir, ir_sils, lidar_depth, lidar_points, pose2d, pose3d, radar4d, radar_depth, rgb, rgb_sils]
```

#### 4. Training & Testing

Train:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py \
  --cfgs configs/singlemodal/rgb_sils/DeepGaitV2.yaml \
  --phase train --log_to_file
```

Test:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=7207 opengait/main.py \
  --cfgs configs/singlemodal/rgb_sils/DeepGaitV2.yaml \
  --phase test --log_to_file
```

You can also use the provided command templates:
- `single_modal.sh`
- `cross_modal.sh`
- `multi_modal.sh`
- `omni_modal.sh`

## Model Zoo
Coming soon.

## Acknowledgement
This codebase is built upon the OpenGait framework and extends it for MMGait tasks.

## Citation
If you find this project useful in your research, please consider citing our paper (BibTeX will be provided after the camera-ready).

## Note
This code is strictly intended for **academic purposes** and must not be used for any form of commercial use.

## Contact
If you have any questions, please contact `BNU-IVC@outlook.com`.
