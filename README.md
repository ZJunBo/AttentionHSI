# Fusing Spatial Attention with Spectral-channel Attention Mechanism for Hyperspectral Image Classification via Encoder-decoder Networks
In this work, we propose an encoder-decoder network that fuses spatial attention and spectral-channel attention for HSI classification.
## Requirements
* python 3.8 
* pytorch 1.9
* cuda 11.1 
* cuDNN 8.0.4
* torchvision 0.10.0
* timm ==0.3.2
* sklearn numpy spectral matplotlib scipy  


## Data preparation
- The project should be organized as:
  ```
  ├── Dataset
  │   ├── IndianPines
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
put Indian_pines KSC PU dataset in ```Datasets``` folder <br>  The dataset can be downloaded from the [[website]](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University)
run the ```Demo_IP_Seg.py``` for Indian Pines dataset training and testing<br> 


