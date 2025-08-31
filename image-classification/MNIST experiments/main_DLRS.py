from __future__ import print_function
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import numpy as np
# import visdom
# vis = visdom.Visdom(env='adaptive_lr')
# win = vis.line(X=[0.], Y=[0.], win='train_loss_ada', opts={'title':'adadelta'})
import os

from utils.dlrs import DLRS

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

        self.lr_upper_threshold = 1e-3
        self.lr_lower_threshold = 5e-7

        self.loss_metrics = []

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


def train(args, model, device, train_loader, optimizer, epoch, lr):
    model.train()
    
    loss_metrics = []
    
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Negative Log likelihood loss
        loss = F.nll_loss(output, target)
        loss.backward()

        loss_c = loss.cpu().detach().numpy()
        loss_metrics.append(loss_c)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        # scheduler.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} \
                  ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            # vis.line([loss.item()], [(epoch - 1)*len(train_loader) + batch_idx],
            #          win='train_loss_adadelta',
            #          opts={'title':'adadelta'},
            #          update= None if (epoch - 1)*len(train_loader) + batch_idx == 0 else 'append')
            # train_loss.append(loss.item())
            if args.dry_run:
                break
    # return train_loss
    return lr, loss_metrics


def test(model, device, test_loader):
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

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)


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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=50, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    del_i = 20.0
    del_d = 1.0

    scheduler = DLRS(0.5, 1e-7, del_i, del_d)
    # train_loss = []
    test_loss = []
    test_acc = []
    begin = time.perf_counter()
    latency_train = 0.
    lr = args.lr

    # train_log_dir = f'logs_exp_3/DLRS/batchsize_{args.batch_size}/lr_{args.lr}/train'
    # train_log_dir = f'mnist_logs/mnist/sgd/dlrs/lr_{args.lr}/train'
    train_log_dir = f'mnist_logs/mnist/sgd/dlrs/batchsize_{args.batch_size}/train'
    writer = SummaryWriter(log_dir=train_log_dir)
    print(f"Initialized Tensorboard logs at {train_log_dir}")
    
    for epoch in range(1, args.epochs + 1):
        latency_train_s = time.perf_counter()
        lr, loss_metrics = train(args, model, device, train_loader, optimizer, epoch, lr)
        writer.add_scalar("LR", lr, epoch)
        
        loss_train = np.mean(loss_metrics)
        writer.add_scalar("Loss/train", loss_train, epoch)

        latency_train_end = time.perf_counter()
        latency_train += (latency_train_end - latency_train_s)
        a, b = test(model, device, test_loader)
        test_loss.append(a)
        writer.add_scalar("Loss/test", test_loss[-1], epoch)
        
        test_acc.append(b)
        writer.add_scalar("Accuracy/test", test_acc[-1], epoch)
        
        lr, _ = scheduler.step_2(lr, loss_metrics)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        writer.flush()
    end = time.perf_counter()
    
    writer.close()

    with open('./results_exp.txt', 'a+') as fl:
        fl.write(
            f'\nOptimizer: SGD,\n\
                Scheduler: DLRS,\n\
                Seed: {args.seed},\n\
                LR: {args.lr},\n\
                batch size: {args.batch_size}\n\
                test loss: {test_loss}\n\
                test acc: {test_acc}\n\
                total time: {end - begin}\n\n----------------')
    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    print(f"Total time is {end - begin}, training time is {latency_train}")


if __name__ == '__main__':
    main()
