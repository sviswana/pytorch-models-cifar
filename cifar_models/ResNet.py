import torch.nn as nn
import torch.nn.functional as F

# Slight variation of classic ResNet model (depth of 50) for compatibility
# with CIFAR 32 x 32 x 3 images

class IdentityBlock(nn.Module):
    def __init__(self, inchannels, outchannels, stride=1, convBlock=False):
        expansion = 4
        self.convBlock = convBlock
        super(IdentityBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=1, bias = False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias = False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels * expansion, kernel_size=1, bias = False),
            nn.BatchNorm2d(outchannels * expansion),
        )
        self.midBlock = nn.Sequential(
                nn.Conv2d(inchannels, outchannels * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannels * expansion),
            )

    def forward(self, x):
        x_shortcut = x
        if self.convBlock:
            x_shortcut = self.midBlock(x)
        x = self.features(x)
        x += x_shortcut
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, depth=50, num_classes=10):
        super(ResNet, self).__init__()
        block = IdentityBlock
        self.inchannels = 16
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            IdentityBlock(16, 16, convBlock = True),
            IdentityBlock(64, 16),
            IdentityBlock(64, 16),
        )
        self.stage2 = nn.Sequential(
            IdentityBlock(64, 32, stride = 2, convBlock = True),
            IdentityBlock(128, 32),
            IdentityBlock(128, 32),
            IdentityBlock(128, 32),
        )
        self.stage3 = nn.Sequential(
            IdentityBlock(128, 64, stride = 2, convBlock = True),
             IdentityBlock(256, 64),
             IdentityBlock(256, 64),
             IdentityBlock(256, 64),
             IdentityBlock(256, 64),
             IdentityBlock(256, 64),
        )
        self.stage4 = nn.Sequential(
            IdentityBlock(256, 128, convBlock = True),
             IdentityBlock(512, 128),
             IdentityBlock(512, 128),
        )
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
