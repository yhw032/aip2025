import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

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

def plot_tensor(tensor, mode=1, num_col=None, label=None):
    if mode == 1:
        tensor = tensor.detach()  # detach from tensor graph
        tensor = tensor.permute(0, 2, 3, 1)  # (32, 3, 7, 7) -> (32, 7, 7, 3)
        npArr = tensor.cpu().numpy()  # pytorch tensor to numpy arr
        num_row = int(npArr.shape[0] / num_col)
        fig, ax = plt.subplots(num_row, num_col)
        idx = 0
        for r in range(num_row):
            for c in range(num_col):
                ax[r, c].imshow(npArr[idx,])
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
                if label is not None:
                    ax[r, c].set_title('label: {}\npred: {}'.
                                       format(label[0][idx], label[1][idx]), fontsize=5)
                idx+=1
        plt.tight_layout()
        plt.show()
    else:
        tensor = tensor.detach().cpu()  # detach from tensor graph
        img_grid = torchvision.utils.make_grid(tensor, nrow=num_col, padding=2, pad_value=1, normalize=True)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7,7), stride=(1,1))
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
    state_dict = torch.load('./ckpt_0/model_state_99.st')
    model.load_state_dict(state_dict, strict=True)

    weight_tensor = model.conv1.weight
    plot_tensor(weight_tensor, mode=2, num_col=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model.eval()
    with torch.no_grad():
        for step, (data, targets) in enumerate(testloader):
            data = data.to(device, dtype=torch.float)
            output, _ = model(data)
            _, pred = output.topk(1, 1, True, True)
            target_cls = [classes[x] for x in targets.cpu().numpy()]
            pred_cls = [classes[x] for x in pred.cpu().squeeze().numpy()]
            print(target_cls, pred_cls, sep='\n')
            plot_tensor(data, mode=2, num_col=4, label=[target_cls, pred_cls])
            break