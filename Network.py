import torch.nn as nn
import torch.nn.init as init
from cfg import par

class Denoising_Net_gray(nn.Module):
    def __init__(self, depth=15, input_channel=par.input_channel, n_channel=72, output_channel=par.output_channel):
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
    def __init__(self, depth=12, input_channel=par.input_channel, n_channel=108, output_channel=par.output_channel):
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

                
