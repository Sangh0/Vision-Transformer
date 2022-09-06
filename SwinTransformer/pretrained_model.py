import timm

def get_pretrained_model(
    model_name: str = 'swin_tiny_patch4_window7_224',
    pretrained: bool = True,
    num_classes: int = 2,
):
    pretrained_model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return pretrained_model