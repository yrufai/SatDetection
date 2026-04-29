import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(pretrained=True):
    """
    Build U-Net with ResNet-50 encoder.

    Args:
        pretrained: if True, load ImageNet weights (Model B)
                    if False, randomly initialize (Model A / baseline)

    Returns:
        model with 2-channel sigmoid output (buildings, roads)
    """
    encoder_weights = "imagenet" if pretrained else None

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=2,
        activation="sigmoid",
    )

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    import torch

    for pretrained in [False, True]:
        label = "Model B (pretrained)" if pretrained else "Model A (random init)"
        model = build_model(pretrained=pretrained)
        total, trainable = count_parameters(model)
        print(f"{label}: {total:,} total params, {trainable:,} trainable")

        dummy = torch.randn(2, 3, 256, 256)
        out = model(dummy)
        print(f"  Output shape: {out.shape}")  # expect (2, 2, 256, 256)
        print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
