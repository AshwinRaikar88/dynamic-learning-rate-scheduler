'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 0 or last checkpoint epoch

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
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = SimpleDLA()
# net = VGG('VGG19')
# net = ResNet18()
# net = GoogLeNet()
# net = MobileNetV2()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
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
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
optimizer = optim.SGD(net.parameters(), lr=args.lr)

def grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def lr_adap(lr, loss_diff, norm):
    new_lr = (lr/2 * norm**2 + loss_diff) / (norm**2)
    return new_lr.data.item()

# Training
def train(epoch, lr, optimizer, losses_neib):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_diff = 0.
    lrlist_inner = []
    lossdiff_list_inner = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        losses_neib[0] = losses_neib[1]
        losses_neib[1] = loss
        loss_diff += (losses_neib[0] - losses_neib[1]).clone().detach()
        # loss_diff += torch.max(torch.tensor([losses[batch_idx % 2] - losses[(batch_idx + 1) % 2], 0]))
        # lr = lr/2 + (loss_diff>0)*lr/2 + (loss_diff<=0)*lr/4
        #lr = 0.005*(epoch<30) + (epoch>30)*lr_adap(lr, loss_diff, grad_norm(net.parameters()))
        # print(lr, loss_diff, grad_norm(net.parameters()))
        # nn.utils.clip_grad_norm_(net.parameters(), 10)
        # print(lr)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(lr, grad_norm(net.parameters()), loss_diff, loss.item(), correct, 100.*correct/total)
        if batch_idx % 20 == 0:
            amplify1 = 1/2 + (torch.atan(torch.as_tensor(loss_diff+1/lr)))/(5*3.14)
            amplify21 = 1/4 - (torch.atan(torch.as_tensor(lr)))/(5*3.14)
            amplify22 = 1/4 - (torch.atan(torch.as_tensor(-loss_diff+1/lr)))/(5.5*3.14)
            lr = lr/2 + (loss_diff > 10**(-5)) * lr * amplify1 + (loss_diff < -10**(-5)) * lr * (amplify21+amplify22)
            print(lr.data.item(), loss_diff.data.item(), grad_norm(net.parameters()))
            optimizer = optim.SGD(net.parameters(), lr=lr)
            lrlist_inner.append(lr.data.item())
            lossdiff_list_inner.append(loss_diff.data.item())
            loss_diff = 0.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return lr.data.item(), lrlist_inner, losses_neib, lossdiff_list_inner

def train1(epoch, optimizer):
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
        # print(lr, grad_norm(net.parameters()), loss_diff, loss.item(), correct, 100.*correct/total)
        if batch_idx % 20 == 0:
            print(lr, grad_norm(net.parameters()))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))

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
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    test_loss = test_loss/(batch_idx+1)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    if acc > best_acc:
        # print('Saving..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return acc, test_loss

lr = args.lr
test_acc = []
test_loss = []
lr_list = []
losses_neib = [0, 0]
lossdiff_list = []
begin = time.perf_counter()
total_epochs = 100
fraction = 0.6
for epoch in range(start_epoch, start_epoch + int(fraction*total_epochs)):
    lr, lrlist_inner, loss_diff_list, lossdiff_list_inner = train(epoch, lr, optimizer, losses_neib)
    a, b = test(epoch)
    test_acc.append(a)
    test_loss.append(b)
    lr_list.append(lrlist_inner)
    lossdiff_list.append(lossdiff_list_inner)
    # scheduler.step()
lr_second_phase = np.mean(lr_list)
print("mean learning rate mean learning rate ", lr_second_phase)
optimizer = optim.SGD(net.parameters(), lr=lr_second_phase)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
for epoch in range(start_epoch, start_epoch+ int((1-fraction)*total_epochs)):
    train1(epoch + int(fraction*total_epochs), optimizer)
    scheduler.step()
    a, b = test(epoch)
    test_acc.append(a)
    test_loss.append(b)

end = time.perf_counter()

with open('./result_adacomp1002.txt', 'a+') as fl:
    fl.write('\n Optimizer is Adacomp, model is SimpleDLA, LR is {}, \n test loss is {}'
             ' \n test acc is {} \n time is {} \n \n'.
                 format(args.lr, test_loss, test_acc, end-begin))