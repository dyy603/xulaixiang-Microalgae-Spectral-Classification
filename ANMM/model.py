# model.py
import torch
import torch.nn as nn
from msa import MultiHeadSelfAttention
from mla import ImageMLA


class AlexNet(nn.Module):
    def __init__(self, num_classes=5, init_weights=False):
        super().__init__()

        # 特征提取层（MSA与MLA协同工作）
        self.features = nn.Sequential(
            # Stage 1 (浅层局部特征)
            nn.Conv2d(3,  48, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, 2),

            # Stage 2 (中级语义特征)
            nn.Conv2d(48, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            # Stage 3 (深层全局建模)
            nn.Conv2d(128, 192, 3, padding=1),
            nn.ReLU(inplace=True),


            # Stage 4
            nn.Conv2d(192, 256, 3, padding=1),
            nn.ReLU(inplace=True),


            # Stage 5 (特征精炼)
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            *self._build_mla_block(256),

            # Stage 6
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            *self._build_msa_block(128),
            nn.MaxPool2d(3, 2),

        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def _build_mla_block(self, channels):
        """确保通道数满足dim % 4 == 0"""
        assert channels % 4 == 0, f"通道数 {channels} 必须能被4整除"
        return [
            ImageMLA(channels,
                    num_heads=4,
                    kv_rank=channels//2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True)
        ]

    def _build_msa_block(self, channels):
        """安全构建MSA模块"""
        embed_dim = channels // 4
        if (embed_dim % 4 != 0) or (embed_dim < 4):
            return []
        return [
            nn.Conv2d(channels, embed_dim, 1),
            nn.ReLU(inplace=True),
            MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=4),
            nn.Conv2d(embed_dim, channels, 1),
            nn.ReLU(inplace=True)
        ]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, ImageMLA):
                # MLA特殊初始化
                nn.init.xavier_uniform_(m.q_proj[0].weight)
                nn.init.zeros_(m.q_proj[2].weight)
                nn.init.xavier_uniform_(m.kv_proj[0].weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                nn.init.zeros_(m.out_proj.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x