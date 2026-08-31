"""Fast and effective MobileNetV2 classifier for FER."""

import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class EmotionResNet18(nn.Module):
    def __init__(self, num_classes: int = 7, freeze_backbone: bool = True):
        super().__init__()
        self.network = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        if freeze_backbone:
            for parameter in self.network.parameters():
                parameter.requires_grad = False

        in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, inputs):
        return self.network(inputs)