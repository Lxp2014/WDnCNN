import numpy as np
import cv2
import pywt
from cfg import par


def patch_generator_color(im, par):
    patch_t = []
    for s in par.scales:
        im_s = cv2.resize(im, (int(im.shape[0]*s), int(im.shape[1]*s)), interpolation=cv2.INTER_CUBIC)
        for i in range(0, im_s.shape[0]-par.patch_size+1, par.stride):
            for j in range(0, im_s.shape[1]-par.patch_size+1, par.stride):
                x = np.divide(im_s[i:i+par.patch_size, j:j+par.patch_size, :], 255).transpose(2, 0, 1)  # C W H
                for k in range(par.aug_time):
                    x_aug = data_aug_color(x, mode=np.random.randint(0, 8))
                    patch_t.append(x_aug)
    return patch_t


def patch_generator_gray(im, par):
    patch_t = []
    for s in par.scales:
        im_s = cv2.resize(im, (int(im.shape[0]*s), int(im.shape[1]*s)), interpolation=cv2.INTER_CUBIC)
        for i in range(0, im_s.shape[0]-par.patch_size+1, par.stride):
            for j in range(0, im_s.shape[1] - par.patch_size + 1, par.stride):
                x = np.divide(im_s[i:i + par.patch_size, j:j + par.patch_size], 255)  # W H
                for k in range(par.aug_time):
                    x_aug = data_aug_gray(x, mode=np.random.randint(0, 8))
                    patch_t.append(x_aug)
    return patch_t


def data_aug_color(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img, axes=(1, 2))
    elif mode == 3:
        return np.flipud(np.rot90(img, axes=(1, 2)))
    elif mode == 4:
        return np.rot90(img, k=2, axes=(1, 2))
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2, axes=(1, 2)))
    elif mode == 6:
        return np.rot90(img, k=3, axes=(1, 2))
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3, axes=(1, 2)))


def data_aug_gray(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def color_tensor_generator(noisy, target):
    tensor_n = []
    tensor_t = []
    for i in range(par.img_channel):
        coeffs_n = pywt.dwt2(noisy[i, :, :], wavelet=par.wave_base)
        cA, (cH, cV, cD) = coeffs_n
        tensor_n.append(cA)
        tensor_n.append(cH)
        tensor_n.append(cV)
        tensor_n.append(cD)

        coeffs_t = pywt.dwt2(target[i, :, :], wavelet=par.wave_base)
        cA, (cH, cV, cD) = coeffs_t
        tensor_t.append(cA)
        tensor_t.append(cH)
        tensor_t.append(cV)
        tensor_t.append(cD)
    return tensor_n, tensor_t


def gray_tensor_generator(noisy, target):
    tensor_n = []
    tensor_t = []
    coeffs_n = pywt.dwt2(noisy, wavelet=par.wave_base)
    cA, (cH, cV, cD) = coeffs_n
    tensor_n.append(cA)
    tensor_n.append(cH)
    tensor_n.append(cV)
    tensor_n.append(cD)

    coeffs_t = pywt.dwt2(target, wavelet=par.wave_base)
    cA, (cH, cV, cD) = coeffs_t
    tensor_t.append(cA)
    tensor_t.append(cH)
    tensor_t.append(cV)
    tensor_t.append(cD)
    return tensor_n, tensor_t


def gray_reconstruction(tensor):
    cA, cH, cV, cD = tensor[0, :, :], tensor[1, :, :], tensor[2, :, :], tensor[3, :, :]
    coeffs = cA, (cH, cV, cD)
    img_denoised = pywt.idwt2(coeffs, wavelet=par.wave_base)
    return img_denoised


def color_reconstruction(tensor):
    img_denoised = []
    for i in range(par.img_channel):
        channel_w = tensor[4*i: 4*(i+1), :, :]
        cA, cH, cV, cD = channel_w[0, :, :], channel_w[1, :, :], channel_w[2, :, :], channel_w[3, :, :]
        coeffs = cA, (cH, cV, cD)
        channel = pywt.idwt2(coeffs, wavelet=par.wave_base)
        img_denoised.append(channel)
    img_denoised = np.stack(img_denoised, axis=0).astype(np.float32)
    return img_denoised


