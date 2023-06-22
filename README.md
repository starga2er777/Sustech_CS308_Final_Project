# Semantic Segmentation based on DeepLab-v2 with a ResNet-101 backbone

Forked from [DeepLab with PyTorch](https://github.com/kazuto1011/deeplab-pytorch)

For further information, see report.  

The new model is not here because of its size. 

To process a single image:

```
python demo.py single \
    --config-path configs/voc12.yaml \
    --model-path our_model.pth \
    --image-path image.jpg
```

To run on a webcam:

```
python demo.py live \
    --config-path configs/voc12.yaml \
    --model-path our_model.pth
```