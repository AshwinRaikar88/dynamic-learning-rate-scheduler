from __future__ import print_function
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR
import numpy as np
from metrics import ConfusionMatrix
# import visdom
# vis = visdom.Visdom(env='adaptive_lr')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def lr_adap(lr, loss_diff, norm):
    new_lr = (lr/2 * norm**2 + loss_diff) / (norm**2 + 1e-6)
    return new_lr.data.item()

losses = [0, 0]

def train(args, model, device, optimizer, train_loader, epoch, lr, losses_neib, lr_list):
    model.train()
    loss_diff = 0.
    loss_metrics = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss_metrics.append(loss.cpu().detach().numpy())
        loss.backward()
        losses_neib[0] = losses_neib[1]
        losses_neib[1] = loss
        # loss_diff += torch.tensor(losses_neib[0] - losses_neib[1])
        loss_diff += (losses_neib[0] - losses_neib[1]).clone().detach()
        amplify1 = 1 / 2 + (torch.atan(torch.as_tensor(loss_diff + 1 / lr))) / (5.0 * 3.14)
        amplify21 = 1 / 4 - (torch.atan(torch.as_tensor(lr))) / (0.6 * 3.14)
        amplify22 = 1 / 4 - (torch.atan(torch.as_tensor(-loss_diff + 1 / lr))) / (5.5 * 3.14)
        lr = lr / 2 + (loss_diff > 10 ** (-5)) * lr * amplify1 + (loss_diff < -10 ** (-5)) * lr * (
                amplify21 + amplify22)
        # print(lr.data.item(), loss_diff.data.item(), grad_norm(model.parameters()))
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # loss_diff = 0.
        # lr = lr_adap(lr, loss_diff, grad_norm(model.parameters()))
        # nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            lr_list.append(lr.data.item())
            loss_diff = 0.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # vis.line([loss.item()], [(epoch - 1)*len(train_loader) + batch_idx],
            #          win='train_loss_compenlr',
            #          opts={'title':'compen','legend':['loss']},
            #          update= None if (epoch - 1)*len(train_loader) + batch_idx == 0 else 'append')
            if args.dry_run:
                break
    return lr, losses_neib, lr_list, loss_metrics

def train1(args, model, device, optimizer, train_loader, epoch):
    model.train()
    loss_diff = 0.
    loss_metrics = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss_metrics.append(loss.cpu().detach().numpy())
        loss.backward()
        # print(lr.data.item(), loss_diff.data.item(), grad_norm(model.parameters()))
        # loss_diff = 0.
        # lr = lr_adap(lr, loss_diff, grad_norm(model.parameters()))
        # nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # vis.line([loss.item()], [(epoch - 1)*len(train_loader) + batch_idx],
            #          win='train_loss_compenlr',
            #          opts={'title':'compen','legend':['loss']},
            #          update= None if (epoch - 1)*len(train_loader) + batch_idx == 0 else 'append')
            if args.dry_run:
                break
        return loss_metrics

def test(model, device, test_loader, epoch, confusion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            confusion.update(pred, target)
    # confusion.plot()
    acc, precision, recall, f1_score = confusion.summary()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # vis.line([test_loss], [epoch],
    #          win='test_loss_compenlr',
    #          opts={'title': 'compen', 'legend': ['test_loss']},
    #          update= 'append')
    # return test_loss, 100. * correct / len(test_loader.dataset)
    return acc, precision, recall, f1_score


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--beta', type=float, default=0.6, metavar='M',
                        help='tuing increment and decrement tradeoff (default: 0.6)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=50, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    num_classes = len(train_loader.dataset.classes)
    label = train_loader.dataset.classes
    confusion = ConfusionMatrix(num_classes=num_classes, labels=label)


    model = Net().to(device)
    lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # train_log_dir = f'logs_adacomp/SGD/batchsize_{args.batch_size}/train'
    train_log_dir = f'mnist_logs/mnist/sgd/adacomp/batchsize_{args.batch_size}/train'
    writer = SummaryWriter(log_dir=train_log_dir)
    print(f"Initialized Tensorboard logs at {train_log_dir}")

    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    test_loss, test_acc, test_precision, test_recall, test_f1 = [],[],[],[],[]

    begin = time.perf_counter()
    latency_train =  0.
    losses_neib = [0, 0]
    lr_list = []
    lossdiff_list = []
    loss_list = []
    for epoch in range(1, int(0.5*args.epochs) + 1):
        latency_train_s = time.perf_counter()
        lr, losses_neib, lr_list ,loss_train= train(args, model, device, optimizer, train_loader, epoch,
                                lr, losses_neib, lr_list)
        latency_train_end = time.perf_counter()
        latency_train += (latency_train_end-latency_train_s)

        train_loss = np.mean(loss_train)

        writer.add_scalar("LR", lr, epoch)

        writer.add_scalar("Loss/train", train_loss, epoch)
        
        acc, precision, recall, f1_score = test(model, device, test_loader, epoch, confusion)
        # test_loss.append(a)
        test_acc.append(acc), test_precision.append(precision)
        writer.add_scalar("Accuracy/test", test_acc[-1]*100, epoch)
        
        test_recall.append(recall), test_f1.append(f1_score)



    lr = np.mean(lr_list)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    for epoch in range(int(0.5*args.epochs) + 1, args.epochs + 1):
        # record training latency
        latency_train_s = time.perf_counter()
        loss_train = train1(args, model, device, optimizer, train_loader, epoch)
        latency_train_end = time.perf_counter()
        latency_train += (latency_train_end-latency_train_s)

        train_loss = np.mean(loss_train)

        writer.add_scalar("LR", lr, epoch)

        writer.add_scalar("Loss/train", train_loss, epoch)

        # record accuracy, precision, recall, f1_score
        acc, precision, recall, f1_score= test(model, device, test_loader, epoch, confusion)
        # # test_loss.append(a)
        test_acc.append(acc), test_precision.append(precision)

        writer.add_scalar("Accuracy/test", test_acc[-1]*100, epoch)

        test_recall.append(recall), test_f1.append(f1_score)
        # scheduler.step()
    end = time.perf_counter()

    with open('./results_adacomp_mnist.txt', 'a+') as fl:
        fl.write('Optimizer is AdaCompen, Seed is {}, LR is {}, batch size is {} \n '
                 ' test acc is{} \n test pre is {} \n test reca is {} \n test f1 is {}'
                 '\n time is {} \n training latency is {}, lr_list is {} \n\n'.
                 format(args.seed, args.lr, args.batch_size,  test_acc, test_precision, test_recall,
                        test_f1, end-begin, latency_train, lr_list))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    print("Total time is {}, training time is {}", end-begin, latency_train)


if __name__ == '__main__':
    main()
