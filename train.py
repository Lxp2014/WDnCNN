import torch as t
import torch.utils.data as data
import torch.optim as optim
from data import train_dataset_color, valid_dataset_color, train_dataset_gray, valid_dataset_gray
from cfg import par
from Loss_Function import myLoss_gray, myLoss_color
from skimage.measure import compare_psnr, compare_ssim
from Network import Denoising_Net_gray, Denoising_Net_color
from utils import gray_reconstruction, color_reconstruction
import warnings

warnings.filterwarnings("ignore")

Validation_color = ['CBSD68', 'Kodak24', 'McMaster']  # 'Kodak24', 'CBSD68', 'McMaster'
Validation_gray  = ['Set12', 'BSD68']  # 'Set12', 'BSD68'


beginner = par.beginner
mode = par.mode
print('Training Mode: ' + mode)


if mode == 'gray':
    print('Validation set: Set12, BSD68')
    t.cuda.set_device(1)
    log = open('./log/log.txt', 'w')
    log.truncate()
    log.close()
    # network
    net = Denoising_Net_gray()
    # loss function
    loss_fn = myLoss_gray()
else:
    print('Validation set: CBSD68, Kodak24, McMaster')
    t.cuda.set_device(0)
    log = open('./log/log_c.txt', 'w')
    log.truncate()
    log.close()
    # network
    net = Denoising_Net_color()
    # loss function
    loss_fn = myLoss_color()

# define optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0005)


for i in range(beginner, beginner+50):

    for group in optimizer.param_groups:

        if (i >= beginner and i < beginner+15):
            group['lr'] = 1e-4
        if (i >= beginner+15 and i < beginner+30):
            group['lr'] = 1e-5
        if (i >= beginner+30 and i < beginner+40):
            group['lr'] = 1e-6
        if (i >= beginner+40 and i < beginner+50):
            group['lr'] = 1e-7

    if mode == 'gray':
        if i != 0 and i == beginner:
            net.load_state_dict(t.load('./model/model_gray_epoch_'+str(beginner)))   # modify model name for resuming training
        net.cuda()
        log = open('./log/log.txt', 'a')
        log.write('\n\nEpoch: %d, Learning_rate: %.2e, batch_size: %d' % (i+1, optimizer.param_groups[0]['lr'], par.bs))
        log.close()
    else:
        if i != 0 and i == beginner:
            net.load_state_dict(t.load('./model/model_gray_epoch_'+str(beginner)))  # modify model name for resuming training
        net.cuda()
        log = open('./log/log_c.txt', 'a')
        log.write('\n\nEpoch: %d, Learning_rate: %.2e, batch_size: %d' % (i + 1, optimizer.param_groups[0]['lr'], par.bs))
        log.close()


    # dataset
    if mode == 'color':
        dataset = train_dataset_color(pic=4744 + 1000)
        data_loader = data.DataLoader(dataset, batch_size=par.bs, num_workers=0, shuffle=True, pin_memory=False)
    elif mode == 'gray':
        dataset = train_dataset_gray(pic=4744 + 400 + 600)
        data_loader = data.DataLoader(dataset, batch_size=par.bs, num_workers=0, shuffle=True, pin_memory=False)
    else:
        print('Please select a training mode: gray or color.\n(Using Default Mode: color)')
        dataset = train_dataset_color(pic=4744 + 1000)
        data_loader = data.DataLoader(dataset, batch_size=par.bs, num_workers=0, shuffle=True, pin_memory=False)

    Loss_epoch = 0

    for batch_idx, (noisy, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = net(noisy.cuda())

        Loss = loss_fn(output, target.cuda())
        Loss.backward()
        optimizer.step()

        if batch_idx % 100 == 99 or batch_idx == 0:
            print('#batch: %4d, Loss: %f' % (batch_idx + 1, Loss / par.bs))
        Loss_epoch = Loss_epoch + Loss / par.bs

    print('~~~~~~~~~~~~~~~~~~~~#epoch: %d, Loss: %.4f~~~~~~~~~~~~~~~~~~~~' % (i + 1, Loss_epoch / (batch_idx+1)))
    if mode == 'gray':
        log = open('./log/log.txt', 'a')
        log.write('\nLoss %.4f' % (Loss_epoch / (batch_idx+1)))
        log.close()
    else:
        log = open('./log/log_c.txt', 'a')
        log.write('\nLoss %.4f' % (Loss_epoch / (batch_idx + 1)))
        log.close()

    # validation
    if mode == 'gray':
        Validation = Validation_gray
        t.save(net.state_dict(), './model/model_gray_epoch_' + str(i + 1))
    else:
        Validation = Validation_color
        t.save(net.state_dict(), './model/model_color_epoch_' + str(i + 1))
    for valid_idx in range(len(Validation)):
        dataset_name = Validation[valid_idx]
        if mode == 'gray':
            valid_set = valid_dataset_gray(root='/home/rick/database/Denoising/testset/', dataset_name=dataset_name)
            valid_loader = data.DataLoader(valid_set, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)
        else:
            valid_set = valid_dataset_color(root='/home/rick/database/Denoising/testset/', dataset_name=dataset_name)
            valid_loader = data.DataLoader(valid_set, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)
        PSNR = 0
        SSIM = 0
        for b_idx, (img_n, img_clean) in enumerate(valid_loader):
            img_denoised = net(img_n.cuda())
            img_denoised = img_denoised.cpu().detach().numpy().squeeze()
            if mode == 'gray':
                img_clean = img_clean.numpy().squeeze()  # W*H
                img_denoised = gray_reconstruction(img_denoised)
                img_denoised = img_denoised[0:img_clean.shape[0],0:img_clean.shape[1]]
                img_denoised = img_denoised.clip(0, 1)
                psnr = round(compare_psnr(img_denoised, img_clean, data_range=1), 3)
                ssim = round(compare_ssim(img_denoised, img_clean, data_range=1, multichannel=False), 3)
            else:
                img_clean = img_clean.numpy().squeeze().transpose(1, 2, 0)  # W*H*C
                img_denoised = color_reconstruction(img_denoised)  # C*W*H
                img_denoised = img_denoised[:, 0:img_clean.shape[0], 0:img_clean.shape[1]]
                img_denoised = img_denoised.transpose(1, 2, 0).clip(0, 1)
                psnr = round(compare_psnr(img_denoised, img_clean, data_range=1), 3)
                ssim = round(compare_ssim(img_denoised, img_clean, data_range=1, multichannel=True), 3)
            PSNR = PSNR + psnr
            SSIM = SSIM + ssim
            if dataset_name == 'Set12':
                print('#pic: %d, PSNR: %.4f, SSIM: %.4f' %(b_idx+1, psnr, ssim))
        PSNR = PSNR/(b_idx+1)
        SSIM = SSIM/(b_idx+1)
        print(dataset_name + ': ave_PSNR: %f, ave_SSIM: %f' % (PSNR, SSIM))
        if mode == 'gray':
            log = open('./log/log.txt', 'a')
            log.write('\n' + dataset_name + ': ave_PSNR: %.4f, ave_SSIM: %.4f' % (PSNR, SSIM))
            log.close()
        else:
            log = open('./log/log_c.txt', 'a')
            log.write('\n' + dataset_name + ': ave_PSNR: %.4f, ave_SSIM: %.4f' % (PSNR, SSIM))
            log.close()
