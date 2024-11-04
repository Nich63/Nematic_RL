import torch
import torch.nn as nn

# Conv class: take in 2*256*256 output 16*64*64
class DownSampleConv(nn.Module):
    def __init__(self):
        super(DownSampleConv, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # norm x inside each layer
        self.norm1 = nn.BatchNorm2d(2)
        # norm x before output to -1, 1
        self.norm2 = nn.BatchNorm2d(32)

    def one_norm(self, x):
        return torch.sigmoid(x)

    def forward(self, x):
        x = self.norm1(x)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.one_norm(x)
        # x = x.squeeze()
        # x = self.flatten(x)
        return x

class UpSampleConv(nn.Module):
    def __init__(self):
        super(UpSampleConv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(8, 2, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        
        # Normalization layers matching the encoder
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(8)
        
    def one_norm(self, x):
        return torch.sigmoid(x)
    
    def forward(self, x):
        # x = x.unsqueeze(0)
        x = self.norm1(x)
        x = self.relu(self.deconv1(x))
        x = self.norm2(x)
        x = self.relu(self.deconv2(x))
        x = self.one_norm(x)
        return x


if __name__ == '__main__':
    conv = DownSampleConv()
    input = (torch.randn(256, 256), torch.randn(256, 256))
    input = torch.stack(input)
    input = input.unsqueeze(0)
    print(input.shape)
    output = conv(input)
    print(output.shape)
    decode = UpSampleConv()
    output = decode(output)
    print(output.shape)
    # check the min and max
    # print(output.min())
    # print(output.max())