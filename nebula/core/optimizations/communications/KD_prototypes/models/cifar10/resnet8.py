import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock


class ResNet8(ResNet):
    def __init__(self, num_classes=10):
        # Definimos el número de bloques en cada capa
        layers = [1, 1, 1, 0]  # [layer1, layer2, layer3, layer4]
        super(ResNet8, self).__init__(block=BasicBlock, layers=layers)

        # Reemplazamos la capa fully connected
        # 256 porque ya no coincide con fc.in_features, ya que falta la layer4
        self.fc = nn.Linear(256, num_classes)

    def _forward_impl(self, x):
        # Implementación forward estándar de ResNet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
