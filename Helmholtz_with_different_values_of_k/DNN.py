'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：DNN.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/17 下午6:40 
'''
import torch.optim
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a

class Net_attention(nn.Module):
    def __init__(self, layers):
        super(Net_attention, self).__init__()
        self.layers = layers
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.attention1 = nn.Linear(layers[0], layers[1])
        self.attention2 = nn.Linear(layers[0], layers[1])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)
        nn.init.xavier_normal_(self.attention1.weight.data, gain=1.0)
        nn.init.zeros_(self.attention1.bias.data)
        nn.init.xavier_normal_(self.attention2.weight.data, gain=1.0)
        nn.init.zeros_(self.attention2.bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        encoder_1 = self.activation(self.attention1(x))
        encoder_2 = self.activation(self.attention2(x))
        a = a * encoder_1 + (1 - a) * encoder_2
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
            a = a * encoder_1 + (1 - a) * encoder_2
        a = self.linear[-1](a)
        return a