import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.init as init
import cv2
import pywt
import os
import os.path
from skimage.measure import compare_psnr, compare_ssim
import warnings
warnings.filterwarnings("ignore")


class Denoising_Net_gray(nn.Module):
    def __init__(self, depth=15, input_channel=2, n_channel=72, output_channel=4):
        super(Denoising_Net_gray, self).__init__()
        layers = []
        for _ in range(depth):
            layers.append(
                nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channel, output_channel, kernel_size=(3, 3), padding=(1, 1), bias=False))
        self.denoisingNet = nn.Sequential(*layers)

        self.InputNet0 = nn.Sequential(
            nn.Conv2d(input_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )
        self.InputNet1 = nn.Sequential(
            nn.Conv2d(input_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )
        self.InputNet2 = nn.Sequential(
            nn.Conv2d(input_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )
        self.InputNet3 = nn.Sequential(
            nn.Conv2d(input_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )

        self._initialize_weights()

    def forward(self, x):

        x0 = self.InputNet0(x[:, [0, 4], :, :])
        x1 = self.InputNet1(x[:, [1, 4], :, :])
        x2 = self.InputNet2(x[:, [2, 4], :, :])
        x3 = self.InputNet3(x[:, [3, 4], :, :])

        x = x0 + x1 + x2 + x3

        z = self.denoisingNet(x)
        return x[:, 0:4, :, :] - z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_uniform_(m.weight)
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class Denoising_Net_color(nn.Module):
    def __init__(self, depth=12, input_channel=4, n_channel=108, output_channel=12):
        super(Denoising_Net_color, self).__init__()
        layers = []
        for _ in range(depth):
            layers.append(
                nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channel, output_channel, kernel_size=(3, 3), padding=(1, 1), bias=False))
        self.denoisingNet = nn.Sequential(*layers)

        self.InputNet0 = nn.Sequential(
            nn.Conv2d(input_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )
        self.InputNet1 = nn.Sequential(
            nn.Conv2d(input_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )
        self.InputNet2 = nn.Sequential(
            nn.Conv2d(input_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )
        self.InputNet3 = nn.Sequential(
            nn.Conv2d(input_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(inplace=True)
        )

        self._initialize_weights()

    def forward(self, x):

        x0 = self.InputNet0(x[:, [0, 4, 8, 12], :, :])
        x1 = self.InputNet1(x[:, [1, 5, 9, 12], :, :])
        x2 = self.InputNet2(x[:, [2, 6, 10, 12], :, :])
        x3 = self.InputNet3(x[:, [3, 7, 11, 12], :, :])

        x = x0 + x1 + x2 + x3

        z = self.denoisingNet(x)
        return x[:, 0:12, :, :] - z

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.xavier_uniform_(m.weight)
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                # init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

def color_tensor_generator(noisy):
    tensor_n = []
    for i in range(3):
        coeffs_n = pywt.dwt2(noisy[i, :, :], wavelet='dmey')
        cA, (cH, cV, cD) = coeffs_n
        tensor_n.append(cA)
        tensor_n.append(cH)
        tensor_n.append(cV)
        tensor_n.append(cD)
    return tensor_n

def gray_tensor_generator(noisy):
    tensor_n = []
    coeffs_n = pywt.dwt2(noisy, wavelet='dmey')
    cA, (cH, cV, cD) = coeffs_n
    tensor_n.append(cA)
    tensor_n.append(cH)
    tensor_n.append(cV)
    tensor_n.append(cD)
    return tensor_n

def gray_reconstruction(tensor, wave_base):
    cA, cH, cV, cD = tensor[0, :, :], tensor[1, :, :], tensor[2, :, :], tensor[3, :, :]
    coeffs = cA, (cH, cV, cD)
    img_denoised = pywt.idwt2(coeffs, wavelet=wave_base).clip(0,1)
    return img_denoised

def color_reconstruction(tensor, wave_base):
    img_denoised = []
    for i in range(3):
        channel_w = tensor[4*i: 4*(i+1), :, :]
        cA, cH, cV, cD = channel_w[0, :, :], channel_w[1, :, :], channel_w[2, :, :], channel_w[3, :, :]
        coeffs = cA, (cH, cV, cD)
        channel = pywt.idwt2(coeffs, wavelet=wave_base)
        img_denoised.append(channel)
    img_denoised = np.stack(img_denoised, axis=0).astype(np.float32).clip(0,1) # C*W*H
    return img_denoised.transpose(1, 2, 0)

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(files)


# *************************************
#      For Real World Noise Removal
# Audrey_Hepburn.jpg (Noise_sigma = 10)
# Bears.png          (Noise_sigma = 15)
# Dog.png            (Noise_sigma = 28)
# Flowers.png        (Noise_sigma = 70)
# Frog.png           (Noise_sigma = 15)
# Movie.png          (Noise_sigma = 12)
# Pattern1.png       (Noise_sigma = 12)
# Pattern2.png       (Noise_sigma = 40)
# Pattern3.png       (Noise_sigma = 25)
# Postcards.png      (Noise_sigma = 15)
# Singer.png         (Noise_sigma = 30)
# Stars.png          (Noise_sigma = 18)
# Window.png         (Noise_sigma = 15)
#
# Building.png       (Noise_sigma = 20)
# Chupa_Chups.png    (Noise_sigma = 10)
# David_Hilbert.png  (Noise_sigma = 15)
# Marilyn.png        (Noise_sigma = 7)
# Old_Tom_Morris.png (Noise_sigma = 15)
# Vinegar.png        (Noise_sigma = 20)
# **************************************


sigma = 15  # default
wave_base = 'dmey'
t.cuda.set_device(0)

root = './testset/'
Test_Set = input('Please select a testing set: RNI6 or RNI15?\n')
Dir = Test_Set  # [BSD68, CBSD68, Kodak24, McMaster, RNI15, RNI6, Set12]
Files = file_name(os.path.join(root, Dir))
Test_Img = input('Please select a test image:\n')
img_name = Test_Img
img_dir = os.path.join(root, Dir, img_name)
img_clean = cv2.imread(img_dir, -1).astype(np.float32)
img_clean = np.divide(img_clean, 255)
if Dir == 'RNI15':
    img_n = img_clean
    if Test_Img == 'Audrey_Hepburn.jpg':  sigma = 10
    elif Test_Img == 'Bears.png':         sigma = 15
    elif Test_Img == 'Dog.png':           sigma = 28
    elif Test_Img == 'Flowers.png':       sigma = 70
    elif Test_Img == 'Frog.png':          sigma = 15
    elif Test_Img == 'Movie.png':         sigma = 12
    elif Test_Img == 'Pattern1.png':      sigma = 12
    elif Test_Img == 'Pattern2.png':      sigma = 40
    elif Test_Img == 'Pattern3.png':      sigma = 25
    elif Test_Img == 'Postcards.png':     sigma = 15
    elif Test_Img == 'Singer.png':        sigma = 30
    elif Test_Img == 'Stars.png':         sigma = 18
    elif Test_Img == 'Window.png':        sigma = 15
elif Dir == 'RNI6':
    img_n = img_clean
    if Test_Img == 'Building.png':        sigma = 20
    elif Test_Img == 'Chupa_Chups.png':   sigma = 10
    elif Test_Img == 'David_Hilbert.png': sigma = 15
    elif Test_Img == 'Marilyn.png':       sigma = 7
    elif Test_Img == 'Old_Tom_Morris.png':sigma = 15
    elif Test_Img == 'Vinegar.png':       sigma = 20
else:
    sigma = int(input('Please select a noise level in [0, 75]:\n'))
    np.random.seed(seed=0)  # for reproducibility
    noise = np.random.normal(0, sigma / 255.0, img_clean.shape)  # sigma setting
    img_n = img_clean + noise
    img_n = img_n.astype(np.float32).squeeze()

if len(img_clean.shape) == 2:
    net = Denoising_Net_gray()
    net.load_state_dict(t.load('./model/WDnCNN_model_gray'))
    net.cuda()
    tensor_n = gray_tensor_generator(img_n)
    nlm = np.tile(sigma / 255.0, tensor_n[0].shape).astype(np.float32)  # noise level map
    tensor_n.append(nlm)
    noisy = np.expand_dims(np.stack(tensor_n, axis=0), axis=0)
    noisy = t.from_numpy(noisy.copy())
    img_denoised = net(noisy.cuda())
    img_denoised = img_denoised.cpu().detach().numpy().squeeze()
    img_denoised = gray_reconstruction(img_denoised, wave_base)
    img_denoised = img_denoised[0:img_clean.shape[0], 0:img_clean.shape[1]]
    if Dir != 'RNI6':
        psnr = round(compare_psnr(img_denoised, img_clean, data_range=1), 3)
        ssim = round(compare_ssim(img_denoised, img_clean, data_range=1, multichannel=False), 3)
        print('PSNR: %.4f, SSIM: %.4f' % (psnr, ssim))
else:
    net = Denoising_Net_color()
    net.load_state_dict(t.load('./model/WDnCNN_model_color'))
    net.cuda()
    tensor_n = color_tensor_generator(img_n.transpose(2, 0, 1))
    nlm = np.tile(sigma / 255.0, tensor_n[0].shape).astype(np.float32)  # noise level map
    tensor_n.append(nlm)
    noisy = np.expand_dims(np.stack(tensor_n, axis=0), axis=0)
    noisy = t.from_numpy(noisy.copy())
    img_denoised = net(noisy.cuda())
    img_denoised = img_denoised.cpu().detach().numpy().squeeze()
    img_denoised = color_reconstruction(img_denoised, wave_base)
    img_denoised = img_denoised[0:img_clean.shape[0], 0:img_clean.shape[1], :]
    if Dir != 'RNI15':
        psnr = round(compare_psnr(img_denoised, img_clean, data_range=1), 3)
        ssim = round(compare_ssim(img_denoised, img_clean, data_range=1, multichannel=True), 3)
        print('Test Image: '+img_name+' Noise Level: %d, PSNR: %.4f, SSIM: %.4f' % (sigma, psnr, ssim))

cv2.imwrite('./denoised_figs/noisy.png', np.round(img_n*255), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
cv2.imwrite('./denoised_figs/original.png', np.round(img_clean*255), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
cv2.imwrite('./denoised_figs/denoised.png', np.round(img_denoised*255), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

