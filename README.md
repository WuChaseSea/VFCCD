# VFCCD

Official implementation for the paper "A Remote Sensing Image Change Detection Network with Feature Constraints from Visual Foundation Model"

## Usage

### Install

* Create a conda virtual environment and activate it:

```sh
conda create -n vfccd python=3.8 -y
conda activate vfccd
pip install -r requirements.txt
```

### Data Preparation

Dataset download link:

* WHUCD [https://gpcv.whu.edu.cn/data/building_dataset.html](https://gpcv.whu.edu.cn/data/building_dataset.html)
* CLCD [https://github.com/liumency/CropLand-CD](https://github.com/liumency/CropLand-CD)
* OSCD [https://rcdaudt.github.io/oscd/](https://rcdaudt.github.io/oscd/)

After downloading and processing, move it to ./data folder. This folder path can be modified in corresponding config file.

### Model Preparation

Pretrained model file download link:

CLIP [https://github.com/open-mmlab/mmpretrain/tree/main/configs/clip](https://github.com/open-mmlab/mmpretrain/tree/main/configs/clip)

Move pretrained model file to ./pretraned_models, this folder path can also be modified in corresponding config file.

## Inference

```sh
python application/cd_application.py --config application/predict_pred.yaml
```

## Train

```sh
python Solve.py --config configs/vfm/cd_mmseg_clcd.yaml --gpus 0
```
