import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import shutil
import pandas as pd
import cornet


parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')

parser.add_argument('--initialization', '-i', metavar='INIT',
                    choices=['0', '1', '2', '3','4'])

parser.add_argument('--category', '-ca', metavar='CATEG')

parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")


__spec__ = None

args = parser.parse_args()

transform = transforms.Compose([
         #transforms.Grayscale(num_output_channels=3),
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])


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

# Define custom __getitem__ method for dataloader
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self,index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # image file path
        path = self.imgs[index][0]

        # make new tuple that includes original and path
        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path


#data_dir = 'E:/Projects/2020_lesion/DCNN/Objects/stim/' + args.category
data_dir = 'E:/Projects/2020_lesion/stim/' + args.category
image_datasets = ImageFolderWithPaths(os.path.join(data_dir), transform)
testloader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=False)
dataset_sizes = len(image_datasets)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(model, model_name, criterion, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    all_epoch_loss=[]
    all_epoch_acc=[]

    pred=[]
    true=[]
    pred_wrong=[]
    true_wrong=[]
    image=[]
    im=[]

    for epoch in range(num_epochs):

        model.eval()

        running_loss = 0.0
        running_corrects = 0

        # iterate over data
        for inputs, labels, paths in testloader:
            #print(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() /dataset_sizes
            running_corrects += torch.sum(preds == labels.data)

            # image identity
            preds = preds.cpu().numpy()
            preds = np.reshape(preds,(len(preds),1))
            target = labels.cpu().numpy()
            target = np.reshape(target,(len(preds),1))
            data = inputs.cpu().numpy()

            for i in range(len(preds)):
                pred.append(preds[i])
                true.append(target[i])
                im.append(paths[i])
                if (preds[i] != target[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(target[i])
                    image.append(paths[i])

        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects.double() / dataset_sizes

        #print(epoch_loss)
        #print(epoch_acc)
        #print('Loss: {:.4f} Acc: {:.4f}'.format(
        #    epoch_loss, epoch_acc))
        print('{:.4f}'.format(epoch_acc))

        #all_epoch_loss.append(round(epoch_loss, 4))
        #all_epoch_acc.append(round(epoch_acc, 4))

        all_epoch_loss.append(epoch_loss)
        all_epoch_acc.append(epoch_acc)

    time_elapsed = time.time() - since

    #print('Evaluation complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))

    all_statistics = pd.DataFrame(np.column_stack([all_epoch_loss, all_epoch_acc]),
                                    columns=['loss', 'accuracy'], index=None)
    all_statistics.to_csv(model_name + '_performance.tsv', sep='\t', encoding='utf-8')

    data2 = {
            'imageID': image,
            'pred': pred_wrong,
            'ground': true_wrong}

    imageIDs = pd.DataFrame(data=data2)
    imageIDs.to_csv(model_name + '_' + args.category + '_imageIDs.tsv', sep='\t', encoding='utf-8')

    data3 = {
            'imageID': im,
            'pred': pred,
            'ground': true}

    imageIDs = pd.DataFrame(data=data3)
    imageIDs.to_csv(model_name+"_"+args.initialization + '_' + args.category + '_all.tsv', sep='\t', encoding='utf-8')

# load model
if args.arch == 'resnet6':
    import math
    model_ft = ResNet6(noresBasicBlock, [1,1,1,1])
    model_ft = torch.nn.DataParallel(model_ft)
    num_ftrs = model_ft.module.fc.in_features
    model_ft.module.fc = nn.Linear(num_ftrs,2)

elif args.arch == 'resnet10':
    model_ft = ptcv_get_model('resnet10')
    num_ftrs = model_ft.output.in_features
    model_ft.output = nn.Linear(num_ftrs,2)
    #model_conv.output = nn.Linear(512,1000)
elif args.arch[0:3] == 'cor':
    model = getattr(cornet, args.arch)
    model_ft = model()
    num_ftrs = model_ft.module.decoder.linear.in_features
    model_ft.module.decoder.linear = nn.Linear(num_ftrs,2)
else:
    model_ft = models.__dict__[args.arch]()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,2)

model_name = args.arch
initialization = args.initialization

# load weights from finetuned model
resdir = 'E:/Projects/2020_lesion/DCNN/'#Networks_Neuropsychologia/'
checkpoint = torch.load(resdir+args.arch+"_"+args.category+"_"+args.initialization+'_model_best.pth.tar', map_location='cpu')
model_ft.load_state_dict(checkpoint['state_dict'])
print(model_ft)

# freeze network except for the last layer
#for param in model_conv.parameters():
#    param.requires_grad=False

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()

test_model(model_ft, model_name, criterion, num_epochs=1)
