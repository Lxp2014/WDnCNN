#  [WDnCNN: ENHANCEMENT OF A CNN-BASED DENOISER BASED ON SPATIAL AND SPECTRAL ANALYSIS](https://github.com/RickZ1010/WDnCNN-ENHANCEMENT-OF-A-CNN-BASED-DENOISER-BASED-ON-SPATIAL-AND-SPECTRAL-ANALYSIS "悬停显示")
This is the implementation of the following paper:

**ENHANCEMENT OF A CNN-BASED DENOISER BASED ON SPATIAL AND SPECTRAL ANALYSIS**

*Rui Zhao, Daniel P.K. Lun and Kin-Man Lam*

Abstract: Convolutional neural network (CNN)-based image denoising methods have been widely studied recently, because of their high-speed processing capability and good visual quality. However, most of the existing CNN-based denoisers learn the image prior from the spatial domain, and suffer from the problem of spatially variant noise, which limits their performance in real-world image denoising tasks. In this paper, we propose a discrete wavelet denoising CNN (WDnCNN), which restores images corrupted by various noise with a single model. Since most of the content or energy of natural images resides in the low-frequency spectrum, their transformed coefficients in the frequency domain are highly imbalanced. To address this issue, we present a band normalization module (BNM) to normalize the coefficients from different parts of the frequency spectrum. Moreover, we employ a band discriminative training (BDT) criterion to enhance the model regression. We evaluate the proposed WDnCNN, and compare it with other state-of-the-art denoisers. Experimental results show that WDnCNN achieves promising performance in both synthetic and real noise reduction, making it a potential solution to many practical image denoising applications.

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

### Real-world Denoising Benchmark
We also evaluate our method on the 1,000 cropped real-world noisy images from Darmstadt Noise Dataset. You can find this benchmark at [DND](https://noise.visinf.tu-darmstadt.de/). For denoising the real-world noisy images in DND, we further fine tune our model on PolyU-Real-World-Noisy-Images-Dataset [PRWNID](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset). In the fine tuning, we adopt the sub-network for noisy level estimation in [CBDNet](https://github.com/GuoShi28/CBDNet), and jointly fine tune the sub-network with our WDnCNN.

You can find our results as WDnCNN+ on the [DND official website](https://noise.visinf.tu-darmstadt.de/benchmark/#overview). We achieve **38.87dB** on sRGB images.

One PyTorch Implementation of CBDNet can be found at [CBDNet_PyTorch](https://github.com/IDKiro/CBDNet-pytorch).

## Contact
If you have questions, problems with the code, or find a bug, please let us know. Contact Rui Zhao at rui.zhao16@alumni.imperial.ac.uk  
Thank you!

## Citation

    @INPROCEEDINGS{8804295, 
        author={R. {Zhao} and K. {Lam} and D. P. K. {Lun}}, 
        booktitle={2019 IEEE International Conference on Image Processing (ICIP)}, 
        title={Enhancement of a CNN-Based Denoiser Based on Spatial and Spectral Analysis}, 
        year={2019}, 
        volume={}, 
        number={}, 
        pages={1124-1128}, 
        keywords={Image denoising;convolutional neural networks;spatial-spectral analysis;discrete wavelet transform}, 
        doi={10.1109/ICIP.2019.8804295}, 
        ISSN={2381-8549}, 
        month={Sep.},}
