import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import NoisyLinear
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AtariBody(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5):
        super(AtariBody, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy=noisy

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        return x
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)

    def sample_noise(self):
        pass


class SimpleBody(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5):
        super(SimpleBody, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy=noisy

        self.fc1 = nn.Linear(input_shape[0], 128) if not self.noisy else NoisyLinear(input_shape[0], 128, sigma_init)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

    def feature_size(self):
        return self.fc1(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def sample_noise(self):
        if self.noisy:
            self.fc1.sample_noise()


class TetrisBody(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5):
        super(TetrisBody, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy=noisy

        self.conv_layers = nn.Sequential(
                nn.Conv2d(self.input_shape[0], 32, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, stride=1),
                nn.LeakyReLU(inplace=True),
        )

        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)

        return x
    
    def feature_size(self):
        with torch.no_grad():
            return self.conv_layers(torch.zeros(1, *self.input_shape)).reshape(1, -1).size(1)

    def sample_noise(self):
        pass

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        if in_channels == out_channels:
            self.skip = nn.Sequential()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x) + self.skip(x)))

class MultiBranchBlock(nn.Module):
    def __init__(self, in_channels, branch_channels=None, kernel_sizes=None, paddings=None):
        super(MultiBranchBlock, self).__init__()
        self.branches = nn.ModuleList(
            ResidualBlock(in_channels=in_channels, out_channels=branch_channels, kernel_size=kernel_size, padding=padding) 
            for kernel_size, padding in zip(kernel_sizes, paddings)
        )
    
    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        return sum(outs)
        # return torch.cat(outs, dim=1)


class TetrisBodyV2(nn.Module):
    def __init__(self, input_shape, num_actions=None, noisy=False, sigma_init=0.5):
        super(TetrisBodyV2, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy=noisy

        self.blocks = nn.Sequential(
            MultiBranchBlock(in_channels=2, branch_channels=64, kernel_sizes=[(11, 1), (1, 11), (5, 5)], paddings=[(5, 0), (0, 5), (2, 2)]),
            # MultiBranchBlock(in_channels=32, branch_channels=64, kernel_sizes=[(11, 1), (1, 11), (5, 5)], paddings=[(5, 0), (0, 5), (2, 2)]),
            # nn.AdaptiveAvgPool2d(output_size=(12, 6)),
            MultiBranchBlock(in_channels=64, branch_channels=64, kernel_sizes=[(11, 1), (1, 11), (5, 5)], paddings=[(5, 0), (0, 5), (2, 2)]),
            MultiBranchBlock(in_channels=64, branch_channels=64, kernel_sizes=[(11, 1), (1, 11), (5, 5)], paddings=[(5, 0), (0, 5), (2, 2)]),
            nn.AdaptiveAvgPool2d(output_size=(6, 3)),
            # MultiBranchBlock(in_channels=64, branch_channels=64, kernel_sizes=[(11, 1), (1, 11), (5, 5)], paddings=[(5, 0), (0, 5), (2, 2)]),
            nn.Flatten(start_dim=1)
        )
    
    def preprocess(self, x):
        # x0 = torch.zeros(size=(x.size(0), 3, 23, 12), dtype=torch.float32).to(x.device)
        # x0[:, 0:1] = torch.ones(size=(x.size(0), 1, 23, 12), dtype=torch.float32)
        # x0[:, :, 0:22, 1:11] = x
        # x = x0
        x = x[:, 0:1]
        x = torch.cat([x, 1.0 - x], dim=1)
        return x
        
    def forward(self, x):
        x = self.preprocess(x)
        x = self.blocks(x)
        return x
    
    def feature_size(self):
        with torch.no_grad():
            x = torch.zeros(1, *self.input_shape)
            x = self.preprocess(x)
            return self.blocks(x).size(1)

    def sample_noise(self):
        pass

class TetrisBodyV3(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5):
        super(TetrisBodyV3, self).__init__()        

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy=noisy

        self.conv_layers_square = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_layers_row = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=(1,5), stride=1, padding=(0,2)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_layers_col = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=(5,1), stride=1, padding=(2,0)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_layers_fusion = nn.Sequential(
            nn.Conv2d(99, 32, kernel_size=1, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_layers_top = nn.Sequential(   
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        
    def forward(self, x):
        x_s = self.conv_layers_square(x)
        x_c = self.conv_layers_col(x)
        x_r = self.conv_layers_row(x)
        x_f = self.conv_layers_fusion(torch.cat([x_s, x_c, x_r, x], 1))
        x = self.conv_layers_top(x_f) + x_f

        x = torch.flatten(x, 1)
        return x
    
    def feature_size(self):
        with torch.no_grad():
            return self.forward(torch.zeros(1, *self.input_shape)).reshape(1, -1).size(1)

    def sample_noise(self):
        pass