#  [WDnCNN: ENHANCEMENT OF A CNN-BASED DENOISER BASED ON SPATIAL AND SPECTRAL ANALYSIS](https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS "悬停显示")

## Dependencies
Python >= 3.6.5, Pytorch >= 0.4.1, and cuda-9.2.

## Pretrained Models
Two pretrained WDnCNN models, WDnCNN_model_gray and WDnCNN_model_color, are used for evaluating the denoising performance on grayscale images and color images, respectively. Run demo.py to test the WDnCNN for both synthetic and real-world noise removal.

## Network Architecture
![](https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS/blob/master/figs/figure1.png?raw=true)

### Band Discriminative Training(BDT)
Please refer to the paper and the README.txt file.
<div align=center><img width="500" src="https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS/blob/master/figs/Tab1.png?raw=true"/></div>
<div align=center><img width="500" src="https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS/blob/master/figs/figure2.png?raw=true"/></div>

## Results
### Grayscale AWGN Removal
![](https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS/blob/master/figs/Tab2.png?raw=true)
### Color AWGN Removal
<div align=center><img width="500" src="https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS/blob/master/figs/Tab3.png?raw=true"/></div>

### Visual Results on Real-world Noisy Images
![](https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS/blob/master/figs/Flowers_N.png)  |  ![](https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS/blob/master/figs/Flowers_B.png)
:-------------------------:|:-------------------------:
Noisy                      |  CBM3D
![](https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS/blob/master/figs/Flowers_F.png)  |  ![](https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS/blob/master/figs/Flowers_W.png)
FFDNet                     |  Ours

## Citation
    @INPROCEEDINGS{WDNCNN  
    author={R. {Zhao} and K. M. {Lam} and D. P.K. {Lun}},  
    booktitle={2019 26th IEEE International Conference on Image Processing (ICIP)},  
    title={ENHANCEMENT OF A CNN-BASED DENOISER BASED ON SPATIAL AND SPECTRAL ANALYSIS},  
    year={2019},  
    volume={},  
    number={},  
    month={Oct},  
    }
