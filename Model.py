import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.5):
        super(BasicBlock, self).__init__()
        # Define the first convolutional layer (3x3), followed by batch normalization
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Define the second convolutional layer (3x3), followed by batch normalization
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Define the shortcut path: this ensures the dimensions match between the input and output
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Pass input through the first conv-bn-relu sequence
        out = F.relu(self.bn1(self.conv1(x)))

        # Pass through the second conv-bn sequence
        out = self.bn2(self.conv2(out))

        # Add the shortcut connection (either identity or 1x1 conv)
        out += self.shortcut(x)

        # Apply ReLU after adding the shortcut
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, dropout_prob=0.5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Initial convolution layer (7x7, stride 2) followed by batch normalization and max pooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(3)

        # ResNet layers, each consisting of a sequence of blocks
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout_prob=dropout_prob)

        # Global average pooling followed by a fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 256 * block.expansion)
        self.fc2 = nn.Linear(256 * block.expansion, 64 * block.expansion)
        self.fc3 = nn.Linear(64 * block.expansion, num_classes)
        # Dropout before the fully connected layer
        self.fc_dropout = nn.Dropout(p=dropout_prob)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_prob):
        # Create a sequence of blocks with dropout
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_prob))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv-bn-relu + max pooling
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        # Forward pass through ResNet layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global average pooling
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        # Fully connected layers with dropout
        out = F.relu(self.fc1(out))
        out = self.fc_dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc_dropout(out)
        out = self.fc3(out)

        return out

