import timm
from torchsummary import summary

if __name__ == '__main__':
    pretrained_model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=2,
    )

    summary(pretrained_model, (3, 224, 224), device='cpu')