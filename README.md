# Fusing Spatial Attention with Spectral-channel Attention Mechanism for Hyperspectral Image Classification via Encoder-decoder Networks
In this paper, we propose an encoder-decoder network that fuses spatial attention and spectral-channel attention for HSI classification from three public HSI datasets to tackle these issues.
In terms of feature information fusion, a multi-source attention mechanism including spatial and spectral-channel attention is proposed to encode the spatial and spectral multi-channels contextual information.
Moreover, three fusion strategies are proposed to effectively utilize spatial and spectral-channel attention.
They are direct aggregation, aggregation on feature space, and Hadamard product.
In terms of network development, an encoder-decoder framework is employed for hyperspectral image classification.
The encoder is a hierarchical transformer pipeline that can extract long-range context information. 
The decoder consists of suitable upsampling, skip connection, and convolution blocks, which fuse multi-scale features efficiently.

## Preparation
* python 3.8 
* Pytorch 1.9
* cuda 11.1 
* cuDNN 8.0.4
* torchvision 0.10.0
* sklearn 1.0.2
* numpy 1.22.3
* spectral 0.22.4
* matplotlib 3.5.1
* timm 0.3.2
* scipy 1.8.0


## Data preparation
- The file structure should look like:
  ```
  ├── Dataset
  │   ├── IndianPines
  │   │   ├── Indian_pines.mat
  │   │   ├── Indian_pines_corrected.mat
  │   │   └── Indian_pines_gt.mat
  │   ├── KSC
  │   │   ├── KSC.mat
  │   │   └── KSC_gt.mat
  │   └── PaviaU
  │       ├── PaviaU.mat
  │       └── PaviaU_gt.mat
  └── Model
  │   ├── model_IP
  │   ├── model_PU
  │    ........
  ├── ResultImage
  ├── autis.py
  ├── Demo_IP_Seg
  ├── Demo_PU_Seg
  ├── Demo_KSC_Seg
  ├── H_datapy.PY
  ```

## Training and Testing
put HSI dataset in ```Datasets``` folder <br>
run the ```Demo_IP.py``` for Indian Pines dataset training and testing<br> 


