import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=48),
            nn.Linear(in_features=48, out_features=32),
            nn.Linear(in_features=32, out_features=4)
        )

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    input = torch.ones(10, 1, 16, 16)
    net = Net()
    output = net(input)
    print(output.size())
