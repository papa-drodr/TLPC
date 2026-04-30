import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 150  # dataset label


def get_model(model_name: str, freeze_backbone: bool = True) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        model_name: one of 'resnet50_frozen', 'resnet50_finetune', 'vgg16_frozen', 'convnext_finetune'
        freeze_backbone: if True, only FC layers are trained

    Returns:
        nn.Module: model ready for training
    """
    if model_name == "resnet50_frozen":
        return ResNet50Classifier(freeze_backbone=True)
    elif model_name == "resnet50_finetune":
        return ResNet50Classifier(freeze_backbone=False)
    elif model_name == "vgg16_frozen":
        return VGG16Classifier(freeze_backbone=True)
    elif model_name == "convnext_finetune":
        return ConvNeXtClassifier(freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ------------------------
# Experiment 1 & 2: ResNet50
# ------------------------
class ResNet50Classifier(nn.Module):
    """
    ResNet50 with custom FC head for Pokemon classification.
    - freeze_backbone=True : Exp 1 (FC only)
    - freeze_backbone=False: Exp 2 (full fine-tuning)
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze backbone if required
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Replace final FC layer with custom head
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ------------------------
# Experiment 3: VGG16
# ------------------------
class VGG16Classifier(nn.Module):
    """
    VGG16 with custom FC head for Pokemon classification.
    - freeze_backbone=True : Exp 3 (FC only)
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Freeze convolutional layers if required
        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier with custom head
        in_features = 25088  # VGG16 features output: 512 * 7 * 7
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ------------------------
# Experiment 4: ConvNeXt
# ------------------------
class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Tiny with custom FC head for Pokemon classification.
    - freeze_backbone=False: Exp 4 (full fine-tuning)
    ConvNeXt requires full fine-tuning to bridge the domain gap
    between ImageNet and cartoon-style Pokemon images.
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )

        # Freeze backbone if required (not recommended for ConvNeXt)
        if freeze_backbone:
            for param in backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = backbone.classifier[2].in_features  # 768
        backbone.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ------------------------
# Experiment configs
# ------------------------
EXPERIMENTS = {
    "resnet50_frozen": {
        "description": "ResNet50 — Pretrained, FC only (backbone frozen)",
        "freeze_backbone": True,
        "lr": 1e-3,
        "epochs": 20,
    },
    "resnet50_finetune": {
        "description": "ResNet50 — Pretrained, Full fine-tuning",
        "freeze_backbone": False,
        "lr": 1e-4,  # Lower lr for fine-tuning
        "epochs": 20,
    },
    "vgg16_frozen": {
        "description": "VGG16 — Pretrained, FC only (backbone frozen)",
        "freeze_backbone": True,
        "lr": 1e-3,
        "epochs": 20,
    },
    "convnext_finetune": {
        "description": "ConvNeXt-Tiny — Pretrained, Full fine-tuning",
        "freeze_backbone": False,
        "lr": 1e-4,  # Lower lr for fine-tuning
        "epochs": 20,
    },
}


if __name__ == "__main__":
    # Quick sanity check for all models
    x = torch.randn(2, 3, 224, 224)  # Batch of 2 images

    for name, config in EXPERIMENTS.items():
        model = get_model(name)
        out = model(x)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[{name}]")
        print(f"  Output shape : {out.shape}")
        print(f"  Trainable    : {trainable:,} / {total:,} params")
        print()
