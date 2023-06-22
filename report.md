# Semantic Segmentation based on DeepLab-v2 with a ResNet-101 backbone



## 1. Introduction

Semantic segmentation is a computer vision task that aims to assign semantic labels to each pixel in an image, effectively dividing the image into different regions corresponding to different objects or classes. The goal is to achieve a fine-grained understanding of the image by assigning meaningful labels to different regions, such as "person," "car," "building," etc.

The main challenge in semantic segmentation is accurately delineating the boundaries of objects and accurately labeling each pixel within those boundaries. This task is more complex than simple object detection or classification because it requires capturing both the global context and the local details of the image.

Overcoming these challenges requires the development of advanced algorithms and architectures, as well as the availability of large-scale annotated datasets for training. Recent advancements in deep learning, especially convolutional neural networks and encoder-decoder architectures, have significantly improved the performance of semantic segmentation systems. However, ongoing research efforts are focused on addressing the remaining challenges to further advance the state-of-the-art in semantic segmentation.



## 2. Related works

Semantic segmentation has been an active area of research in computer vision, and several notable approaches have been proposed to address the challenges of accurate pixel-level labeling. Here are some related works in the field of semantic segmentation:

1. **Fully Convolutional Networks (FCN)**: FCN is the first work of semantic segmentation. It introduced the concept of end-to-end training for semantic segmentation using fully convolutional neural networks. It replaced the fully connected layers of traditional CNN architectures with convolutional layers to preserve spatial information. FCN achieved state-of-the-art performance by predicting dense pixel-wise class labels.
2. **U-Net**: U-net is used to solve simple problem segmentation of small samples. It follows the same basic principles as FCN. U-Net is an encoder-decoder architecture designed for biomedical image segmentation. It includes skip connections between the encoder and decoder, enabling the model to capture both high-level semantic information and fine-grained details. U-Net has been widely adopted and extended for various segmentation tasks.
3. **DeepLab**: DeepLab introduced dilated convolutions to capture multi-scale contextual information. It employs atrous spatial pyramid pooling (ASPP) to aggregate information at different scales and uses a fully connected conditional random field (CRF) for refining the segmentation results.

In this project, DeepLab-v2 model is selected for semantic segmentation. DeepLab model is different from previous ideas and has two features:

- Due to the loss of position information and the high cost of multi-layer up and down sampling, the method to control the receptive field size is converted to Atrous conv.

- Add CRF(conditional random field) to take advantage of the correlation information between pixels: adjacent pixels, or pixels with similar colors, are more likely to belong to the same class.

Also, Deeplab v2 improves on v1 with the introduction of ASPP(Atrous Spatial Pyramid Pooling), as shown in the figure above. We noticed that Deeplab v1 did not fuse information between different layers after expanding the receptive field using porous convolution. ASPP layer is designed to fuse different levels of semantic information: porous convolution with different expansion rates is selected to process Feature maps. Due to different receptive fields, the information levels obtained are also different. ASPP layer concat these different levels of feature maps to carry out information fusion.

 

## 3. Method



## 4. Experiments

### 4.1 Datasets 

#### 4.1.1 PASCAL VOC 2012

![pascal_voc_2012](pics/pascal_voc_2012.png)

The [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) is a benchmark dataset widely used in computer vision for semantic segmentation tasks. It consists of around 11,530 images, each annotated with pixel-level segmentation masks. The dataset covers 20 + 1(background) common object categories and provides a standardized evaluation protocol for measuring algorithm performance. Pascal VOC 2012 has played a significant role in advancing semantic segmentation research and model development.

#### 4.1.2 COCO-Stuff

![coco_stuff](pics/coco_stuff.png)

The [**Common Objects in COntext-stuff** (COCO-stuff) dataset](https://github.com/nightrome/cocostuff#downloads) is a dataset for scene understanding tasks like semantic segmentation, object detection and image captioning. It is constructed by annotating the original COCO dataset, which originally annotated things while neglecting stuff annotations. There are 164k images in COCO-stuff dataset that span over 172 categories including 80 things, 91 stuff, and 1 unlabeled class.

#### 4.1.3 GTA5 Dataset

![gta5](pics/gta5.png)

The GTA5 dataset contains 24966 synthetic images with pixel level semantic annotation. The images have been rendered using the open-world video game *Grand Theft Auto 5* and are all from the car perspective in the streets of American-style virtual cities. There are 19 semantic classes.

#### 4.1.4 Self-labeled Pictures



### 4.2 Implementation

#### 4.2.1 Data Preprocessing

#### 4.2.2 Model Training





### 4.3 Metrics 

Two major metric methods are adopted for model evaluation. In semantic segmentation, there are four categories of pixel-level prediction, respectfully true positive (TP), true negative (TN), false positive (FP) and false negative (FN):

### 4.4 Experimental Design & Results

#### 4.4.1 Experiment Design

4.4.2 Result Analysis



## 5. Conclusion







### Reference

[1] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*, 2018.

[2] H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In *CVPR*, 2018.

[3] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object Classes (VOC) Challenge. *IJCV*, 2010.



### Contributions

Yuhang Wang (12012208): 

Jianlin Li (12012221): 

Zhangjie Chen (12012524):

