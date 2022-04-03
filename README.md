# AttentionHSI

## Requirements
* python 3.8 
* Pytorch 1.9
* cuda 11.1 
* 其他必要库如timm等参考链接https://github.com/microsoft/Swin-Transformer


## Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

## Training and Testing
put HSI dataset in ```Datasets``` folder <br>
run the ```Demo_IP.py``` for Indian Pines dataset training and testing<br> 



## Implementation
This work is implemented based on Swin-Transformer(https://github.com/microsoft/Swin-Transformer).
