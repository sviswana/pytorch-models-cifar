"""
Script for training models on CIFAR dataset. Custom model implementations
to support CIFAR image size (32x32x3)
"""

import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from cifar_models import *

CHECKPOINT_PATH = "./checkpoints/"
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.00001
NUM_EPOCHS = 150

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    global state
    state["lr"] = 0.1 * (0.1 ** (epoch // 30))
    print("Learning rate: ", state["lr"])
    for param_group in optimizer.param_groups:
        param_group['lr'] = state["lr"]

# Download dataset & perform necessary augmentations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=1)

test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False, num_workers=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("cuda!")

# Define model and optimizer
model = ResNet()
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss().cuda(device)

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

checkpoint_file = os.path.join(CHECKPOINT_PATH, "checkpoint.pth.tar")
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# Load checkpoint.
if not os.path.isfile(checkpoint_file):
    start_epoch = 0
    print("--> No checkpoint directory found... Starting from beginning")
else:
    checkpoint = torch.load(checkpoint_file)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('--> Resuming from epoch..', start_epoch)

state = {"lr":0.1}

def test(epoch):
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print("\n TOP1: ", 100.*correct/total)

def train(epoch):
    model.train()
    adjust_learning_rate(optimizer, epoch)
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, tgt = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, tgt)
        train_loss += loss.item()
        _, predicted = pred.max(1)
        total += tgt.size(0)
        correct += predicted.eq(tgt).sum().item()
        loss.backward()
        optimizer.step()

        # Display
        if batch_idx % 100 == 1:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                100.*correct/total))
    if epoch % 5 == 1:
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    },checkpoint=CHECKPOINT_PATH)


for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    train(epoch)
    test(epoch)
