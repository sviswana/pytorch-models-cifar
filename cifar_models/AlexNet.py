import torch.nn as nn
import torch.nn.functional as F

# Slight variation of classic AlexNet model for compatibility
# with CIFAR 32 x 32 x 3 images

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = F.relu(self.conv5(x), inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        F.dropout(x, p=0.5)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x), inplace=True)
        F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x
