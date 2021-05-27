"""Fred: Train CIFAR10 with PyTorch.

Epoch: 0
 [================================================================>]  Step: 1s633ms | Tot: 1m49s | Loss: 1.797 | Acc: 33.956% (16978/50000) 391/391
 [================================================================>]  Step: 71ms | Tot: 9s672ms | Loss: 1.445 | Acc: 45.800% (4580/10000) 100/100
Saving..

Epoch: 1
 [================================================================>]  Step: 172ms | Tot: 1m42s | Loss: 1.341 | Acc: 51.022% (25511/50000) 391/391
 [================================================================>]  Step: 76ms | Tot: 7s520ms | Loss: 1.193 | Acc: 57.370% (5737/10000) 100/100
Saving..

Epoch: 228
 [================================================================>]  Step: 185ms | Tot: 1m36s | Loss: 0.002 | Acc: 99.992% (49996/50000) 391/391
 [================================================================>]  Step: 75ms | Tot: 7s198ms | Loss: 0.187 | Acc: 95.160% (9516/10000) 100/100

"""
""" Trial 2

Epoch: 221
 [================================================================>]  Step: 67ms | Tot: 54s743ms | Loss: 0.002 | Acc: 100.000% (50000/50000) 391/391 
root-INFO: Number of zero_grads (2867200/5243680)
 [================================================================>]  Step: 26ms | Tot: 2s648ms | Loss: 0.176 | Acc: 95.300% (9530/10000) 100/100 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import time
import os
import argparse
import logging
import sys

from models import *
from utils import progress_bar
# from torchsummary import summary
# from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--zero_grad_mea', default=True, help='monitor the zero grad')
parser.add_argument('--epochs', default=300, type=int, help='assigned running epochs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = AlexNet()
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    net.cuda()
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Logging
if not os.path.exists('logging'):
    os.makedirs('logging')
localtime = time.localtime(time.time())
time_str = str(localtime.tm_mon) + '_' + str(localtime.tm_mday) + '_' + str(localtime.tm_hour) + '_' + str(
    localtime.tm_min)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    filename='./logging/' + time_str + '_log.txt',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler(stream=sys.stdout)
console.setLevel(logging.INFO)  # if as INFO will make the console at INFO level thus no additional stdout
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)s-%(levelname)s: %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logger = logging.getLogger()
logger.addHandler(console)
logging.info('Arguments:')
logging.info(args.__dict__)
print("=== Model ===")
print(net)

# summary(net, input_size=(3, 32, 32), device='cuda')
# with torch.cuda.device(0):
#     macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, print_per_layer_stat=True,
#                                              verbose=True)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # logging.info('Accu: {:.3f}%'.format(100. * correct / total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


def num_zero_error_grad(model):
    """
    Return the number of zero gradients and total number of gradients,
    can only be used with prune_flag = True for now
    """
    if model is None:
        return 0

    zeros, total = 0, 0
    non_zero_indices_list = []
    for module in model.children():
        if isinstance(module, (GradConv2d, GradLinear)):  # comment this line to enable for noPrune
            flat_g = module.error_grad.cpu().numpy().flatten()
            zeros += np.sum(flat_g == 0)
            total += len(flat_g)
            non_zero_indices_list = np.where(flat_g != 0)
        elif isinstance(module, nn.Sequential):
            for layer in module:
                # for layer in bblock:
                    if isinstance(layer, (GradConv2d, GradLinear)):
                        print('yes')
                        flat_g = layer.error_grad.cpu().numpy().flatten()
                        zeros += np.sum(flat_g == 0)
                        total += len(flat_g)
                        non_zero_indices_list = np.where(flat_g != 0)
        else:
            raise ValueError('The modules involved are not registered for this fn')

            # print('zero_grad: {}'.format(zeros))
            # print('total: {}'.format(total))
            # print('debug')
    return int(zeros), int(total), non_zero_indices_list


zero_grads_percentage_list = []
for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    if args.zero_grad_mea:
        curr_zero_grads, num_grads, non_zero_indices = num_zero_error_grad(net)
        logging.info("Number of zero_grads ({}/{})".format(curr_zero_grads, num_grads))
        # print("Non zero indices is {}".format(non_zero_indices))
        grad_per = 100. * curr_zero_grads / num_grads
        zero_grads_percentage_list.append(np.around(grad_per, 2))
    test(epoch)
    scheduler.step()
