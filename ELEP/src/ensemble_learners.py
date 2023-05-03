import torch
import torch.nn as nn

class ensemble_regressor_cnn(nn.Module):
    def __init__(
            self,
            in_channels=1,
            classes=1,
            base_channels=8,
            dropout_flag = False
        ):
            super().__init__()

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=(3,3), stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(base_channels, eps=1e-3)
            self.pool1 = nn.MaxPool2d((1, 5))
            self.conv2 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels*2, kernel_size=(3,3), stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(16, eps=1e-3)
            self.pool2 = nn.MaxPool2d((1, 5))
            self.conv3 = nn.Conv2d(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=(1,3), stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(base_channels*4, eps=1e-3)
            self.pool3 = nn.MaxPool2d((1, 4))
            self.conv4 = nn.Conv2d(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=(1,3), stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(base_channels*8, eps=1e-3)
            self.pool4 = nn.MaxPool2d((1, 4))

            self.fc1 = nn.Linear(3200, 640)
            self.bn5 = nn.BatchNorm1d(640, eps=1e-3)
            self.dropout= nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(640, 80)
            self.bn6 = nn.BatchNorm1d(80, eps=1e-3)
            self.dropout= nn.Dropout(p=0.5)
            self.fc3 = nn.Linear(80, classes)

            self.activation = torch.relu
            self.activation2 = torch.sigmoid

    def forward(self, x):
        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        x = self.pool4(self.activation(self.bn4(self.conv4(x))))
        # print(x.shape)

        x = torch.flatten(x, 1)
        # print(x.shape)

        x = self.activation(self.bn5(self.fc1(x)))
        # x = self.dropout(x)
        x = self.activation(self.bn6(self.fc2(x)))
        # x = self.dropout(x)
        x = self.fc3(x)

        return self.activation2(x)