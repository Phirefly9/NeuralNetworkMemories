from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import faiss
import numpy as np

from collections import OrderedDict

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.index = faiss.IndexFlatL2(50)

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #print(self.conv1.weight.shape)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #print(self.conv2.weight.shape)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        #self.rnn = nn.LSTM(input_size=50, hidden_size=150, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(50, 10)

        self.index_cap = 60000
        self.memory_on = False
        
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        tmp = x.cpu().detach().numpy()
        x = F.dropout(x, training=self.training)
        x = x.unsqueeze(1)
        
        if self.memory_on:
            if self.training:
                self.index.add(tmp)
            if self.index.ntotal > self.index_cap:
                remove_range = np.arange(0, self.index.ntotal-self.index_cap)
                self.index.remove_ids(remove_range)
            _, _, tmp3 = self.index.search_and_reconstruct(tmp, 5)
            tmp3 = torch.as_tensor(tmp3, device=torch.device("cuda"))
            x = torch.cat((x, tmp3), dim=1)
        
        x = torch.mean(x, dim=1)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def activate_memory(self):
        print("MEMORY ON")
        self.memory_on = True

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    
    if epoch == args.memory_epoch: model.activate_memory()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--memory_epoch', type=int, default=-1, help='epoch to activate memory saving on')
    # parser.add_argument('--class_hints', default=False, action='store_true', help='adds class label hints to memories')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr) #, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
