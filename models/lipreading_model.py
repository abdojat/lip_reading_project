import torch
import torch.nn as nn
from torchvision import models

class ResNetEncoder(nn.Module):
    """ResNet-18 without final FC: outputs feature vector per frame."""
    def __init__(self, pretrained: bool = False):
        super().__init__()
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        # remove the classification head
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # up to avgpool
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        feat = self.backbone(x)           # (B, 512, 1, 1)
        feat = feat.flatten(1)            # (B, 512)
        return feat

class LipReadingResNetBiLSTM(nn.Module):
    """
    Input:  (B, T, C, H, W)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 4, lstm_hidden: int = 256, lstm_layers: int = 1, bidirectional: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=False)
        self.lstm = nn.LSTM(
            input_size=self.encoder.out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_out_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)                # (B*T, C, H, W)
        feats = self.encoder(x)                   # (B*T, 512)
        feats = feats.view(b, t, -1)              # (B, T, 512)

        out, _ = self.lstm(feats)                 # (B, T, lstm_out_dim)
        last = out[:, -1, :]                      # take last time step
        logits = self.classifier(last)            # (B, num_classes)
        return logits
