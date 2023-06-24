# Semantic Segmentation based on DeepLab-v2 with a ResNet-101 backbone

This is an implementation of **DeepLab v2** with a **ResNet-101** backbone.

For further information, please see report. 

#### Environment setup:

```yaml
dependencies:
  - click
  - conda-forge::pydensecrf
  - cudatoolkit=10.2
  - matplotlib
  - python=3.6
  - pytorch::pytorch>1.2.0
  - pytorch::torchvision
  - pyyaml
  - scipy
  - tqdm
  - pip:
    - addict
    - black
    - joblib
    - omegaconf
    - opencv-python
    - tensorflow
    - torchnet
```

#### Folder structure

```
.
├── configs
├── data
│   ├── datasets
│   │   ├── coco
│   │   │   └── dataset and stuffs...
│   │   ├── cocostuff
│   │   │   └── dataset and stuffs...
│   │   └── voc12
│   │       └── dataset and stuffs...
│   ├── features
│   ├── models
│   │   ├── coco
│   │   │   └── deeplabv1_resnet101
│   │   │       └── caffemodel
│   │   └── voc12
│   │       └── deeplabv2_resnet101_msc
│   │           ├── caffemodel
│   │           └── train_aug
│   └── scores
│       ├── cocostuff10k
│       │   └── deeplabv2_resnet101_msc
│       │       └── test
│       └── voc12
│           └── deeplabv2_resnet101_msc
│               └── val
├── libs
└── scripts
```

The models are not here because of file size. 

#### Run

To process a single image:

```shell
python demo.py single \
    --config-path configs/voc12.yaml \
    --model-path model.pth \
    --image-path image.jpg
```

To run on a webcam:

```shell
python demo.py live \
    --config-path configs/voc12.yaml \
    --model-path model.pth
```

To process a single video and save output:

```shell
python demo.py video \
    --config-path configs/voc12.yaml \
    --model-path model.pth \
    --video-path video.mp4
    --save-path output.mp4
```

