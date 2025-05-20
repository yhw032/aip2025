import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os

def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        num_cor = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            num_cor.append(correct_k.clone())
            acc.append(correct_k.mul(1/batch_size))
    return acc, num_cor


def init_weights(m, init_type='normal', init_gain=0.02):  # define the initialization function
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7), stride=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3,3), (1,1))
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc_code = nn.Linear(in_features=128, out_features=128)
        self.fc_output = nn.Linear(128, 10)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(init_weights)

    def forward(self, x):
        feature_map = self.conv1(x)
        activated = self.relu(feature_map)
        compressed = self.maxpool(activated)
        x = self.maxpool(self.relu(self.conv2(compressed)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        code = self.dropout(self.relu(self.fc_code(x)))
        output= self.fc_output(code)
        return output, code

if __name__ == '__main__':
    if not os.path.isdir("./ckpt_0"):
        os.makedirs("./ckpt_0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    transform_test = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    model = Network()
    model.to(device)

    params = model.parameters()

    optimizer = optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.001)
    train_criterion = nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(1, 100):
        model.train()
        for step, (data, targets) in enumerate(trainloader):
            data = data.to(device, dtype=torch.float)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs, code = model(data)
            loss = nn.CrossEntropyLoss(reduction='mean')(outputs, targets)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc, _ = accuracy(outputs, targets)
            acc = acc[0].item()

            if step % 10 == 0:
                print('Epoch {} Step {}/{} Loss {:.4f} Accuracy {:.4f}'.format(epoch, step, len(trainloader), loss, acc))

        model.eval()
        total_cor = 0
        total_samples = 0

        with torch.no_grad():
            for step, (data, targets) in enumerate(testloader):
                data = data.to(device, dtype=torch.float)
                targets = targets.to(device)
                outputs, code = model(data)
                _, num_cor = accuracy(outputs, targets)
                num_cor = num_cor[0].item()
                total_samples += data.size(0)
                total_cor += num_cor
            acc = total_cor / total_samples
            print('Epoch {} : Accuracy {:.4f}'.format(epoch, acc))
        path = 'ckpt_0/model_state_%d.st'%(epoch)
        torch.save(model.state_dict(), path)