import torch
import torch.nn as nn
import torchvision.models as models

class MelSpectrogramCNN(nn.Module):
    def __init__(self):
        super(MelSpectrogramCNN, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet50.children())[:-1])  # remove the last layer (classification layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(4096, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(p=0.5)
        self.output_layer = nn.Linear(1024, 3)

    def forward(self, x):
        x = self.features(x)
        x1 = self.avgpool(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.bn1(x1)
        x1 = self.dropout1(x1)
        x2 = self.maxpool(x)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.bn2(x2)
        x2 = self.dropout2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn3(self.fc(x))
        x = self.dropout3(x)
        x = nn.ReLU()(x)
        x = self.output_layer(x)
        return x

# Define the Network
class CovidNet(nn.Module):
    def __init__(self):
        super(CovidNet, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1]) # remove the last layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x