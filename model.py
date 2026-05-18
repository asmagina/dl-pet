import torch
import torch.nn as nn

class DoubleConv(nn.Module):    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)

class UNetModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        
        # encoder 
        self.conv1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # bottleneck
        self.conv3 = DoubleConv(128, 256)
        
        # decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride = 2)
        self.conv4 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
        self.conv5 = DoubleConv(128, 64)
        
        # output
        self.output = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        
        x5 = self.conv3(x4)
        
        x = self.up1(x5)
        x = torch.cat([x, x3], dim=1)
        x = self.conv4(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv5(x)
        
        return self.out(x)
        
        