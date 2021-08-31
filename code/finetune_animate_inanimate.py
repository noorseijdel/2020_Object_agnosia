
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from torch.optim import lr_scheduler

import numpy as np
import torchvision

from torchvision import datasets, models, transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

import matplotlib.pyplot as plt
import time
import os
import copy

import argparse
import shutil
import pandas as pd
import cornet
import math


# Enter as argument - model_name and number of epochs
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))
    #and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet10',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')


args = parser.parse_args()

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class noresBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(noresBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet6(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet6, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AvgPool2d(28, stride=1)
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def resnet6(pretrained=False, **kwargs):
    model = ResNet6(noresBasicBlock, [1, 1, 1, 1], **kwargs)
    return model



def train_model(model, model_name, criterion, optimizer, scheduler, condition, initialization, num_epochs=40):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    all_epoch_loss=[]
    all_epoch_acc=[]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            num_images=0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += inputs.size(0)*loss.item()
                running_corrects += torch.sum(preds == labels.data)
                num_images += inputs.size(0)

            if phase == 'train':
                scheduler.step()

            print(num_images, dataset_sizes[phase])

            epoch_loss = running_loss /num_images
            epoch_acc = running_corrects.double() / num_images

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                all_epoch_loss.append(((epoch_loss)))
                all_epoch_acc.append(((epoch_acc)))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    all_statistics = pd.DataFrame(np.column_stack([all_epoch_loss, all_epoch_acc]),
                                    columns=['loss', 'accuracy'], index=None)
    #all_statistics=pd.DataFrame(all_stats)
    #print(all_statistics)
    all_statistics.to_csv(model_name+'_'+condition+"_"+str(initialization)+'_performance.tsv', sep='\t', encoding='utf-8')
    #with open(model_name+'performance.tsv', 'a') as f:
        #all_statistics.to_csv(f, line_terminator=',', index=False, header=False)

    # load best model weights
    model.load_state_dict(best_model_wts)
    state = {
        'epoch': best_epoch + 1,
        'arch': model_name,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, model_name+'_'+condition+'_'+str(initialization)+'_model_best.pth.tar')

    return model


# train model and save best model
for i in range(5):
    print("initialization ", i)
    condition="animate_inanimate"
    model_name = args.arch
    num_ep = args.epochs
    
    if args.arch == 'resnet6':
        print("=> creating model '{}'".format(args.arch))
        model_ft = ResNet6(noresBasicBlock, [1,1,1,1])
        model_ft = torch.nn.DataParallel(model_ft)
        num_ftrs = model_ft.module.fc.in_features
        model_ft.module.fc = nn.Linear(num_ftrs,2)

    elif args.arch == 'resnet10' or args.arch[0:4] == 'alex':
        model_ft = ptcv_get_model(args.arch,pretrained=True)
        num_ftrs = model_ft.output.in_features
        model_ft.output = nn.Linear(num_ftrs,2)
    elif args.arch[0:3] == 'cor':
        model = getattr(cornet, args.arch)
        model_ft = model(pretrained=True)
        num_ftrs = model_ft.module.decoder.linear.in_features
        model_ft.module.decoder.linear = nn.Linear(num_ftrs,2)
    else:
        model_ft = models.__dict__[args.arch](pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    data_dir = 'E:/Projects/Imagesets/MS/' +condition
    print(condition)

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, drop_last=True,
                                             shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(class_names)

    model_ft = train_model(model_ft, model_name, criterion, optimizer_ft, exp_lr_scheduler, condition, num_epochs=num_ep,initialization=i)

