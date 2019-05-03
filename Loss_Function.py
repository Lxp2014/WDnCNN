import torch as t
import torch.nn as nn
import numpy as np
from cfg import par

class myLoss_gray(nn.Module):
    def __init__(self):
        super(myLoss_gray, self).__init__()
        self.lfc = nn.MSELoss(reduction='sum')

    def forward(self, output, target):
        mask = np.ones((par.bs, par.output_channel, output.shape[2], output.shape[3]), dtype=np.float32)
        mask[:, 0, :, :] = 1.5 * mask[:, 0, :, :]   # mu_k for LL band
        mask[:, 1, :, :] = 2.5 * mask[:, 1, :, :]   # mu_k for LH band
        mask[:, 2, :, :] = 2.5 * mask[:, 2, :, :]   # mu_k for HL band
        mask[:, 3, :, :] = 5.0 * mask[:, 3, :, :]   # mu_k for HH band

        mask = t.from_numpy(mask.copy()).cuda()
        loss = self.lfc(output * mask, target * mask).div_(2)
        return loss


class myLoss_color(nn.Module):
    def __init__(self):
        super(myLoss_color, self).__init__()
        self.lfc = nn.MSELoss(reduction='sum')

    def forward(self, output, target):
        mask = np.ones((par.bs, par.output_channel, output.shape[2], output.shape[3]), dtype=np.float32)
        mask[:, [0, 4, 8], :, : ] = 1.5 * mask[:, [0, 4, 8], :, :]   # mu_k for LL band
        mask[:, [1, 5, 9], :, : ] = 2.5 * mask[:, [1, 5, 9], :, :]   # mu_k for LH band
        mask[:, [2, 6, 10], :, :] = 2.5 * mask[:, [2, 6, 10], :, :]  # mu_k for HL band
        mask[:, [3, 7, 11], :, :] = 5.0 * mask[:, [3, 7, 11], :, :]  # mu_k for HH band

        mask = t.from_numpy(mask.copy()).cuda()
        loss = self.lfc(output * mask, target * mask).div_(2)
        return loss




