from .ace2p import ACE2P, AugmentedCE2PHead
from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3, DeepLabV3Edge, DeepLabHeadV3PlusEdge
from .backbone import resnet
from .backbone import mobilenetv2
from .backbone import vits


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone, ace2p=False, use_abn=False):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    if use_abn:
        import functools
        from .modules.bn import InPlaceABNSync
        BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    else:
        from torch import nn
        BatchNorm2d = nn.BatchNorm2d

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation, ace2p=ace2p, norm_layer=BatchNorm2d)

    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3plusedgev1':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3PlusEdge(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    elif name == 'ACE2P':
        return_layers = {'layer4': 'out', 'layer3': 'high_level', 'layer2': 'mid_level', 'layer1': 'low_level'}
        classifier = AugmentedCE2PHead(inplanes, low_level_planes, num_classes, aspp_dilate, use_abn=use_abn)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    if 'edgev1' in name and 'deeplabv3' in name:
        model = DeepLabV3Edge(backbone, classifier)
    elif 'deeplabv3' in name:
        model = DeepLabV3(backbone, classifier)
    elif name == 'ACE2P':
        model = ACE2P(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name == 'deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_vit(name, backbone_name, num_classes, patch_size=16, pretrained_backbone=True, ace2p=False, use_abn=False):
    # if output_stride == 8:
    #     replace_stride_with_dilation = [False, True, True]
    #     aspp_dilate = [12, 24, 36]
    # else:
    #     replace_stride_with_dilation = [False, False, True]
    #     aspp_dilate = [6, 12, 18]

    if use_abn:
        import functools
        from .modules.bn import InPlaceABNSync
        BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    else:
        from torch import nn
        BatchNorm2d = nn.BatchNorm2d

    backbone = vits.__dict__[backbone_name](
            patch_size=patch_size,
            drop_path_rate=0.1,  # stochastic depth
        )
    import torch
    from torchvision.transforms.functional import pil_to_tensor, to_pil_image
    from PIL import Image
    from .backbone.utils import MultiCropWrapper
    chk = torch.load("dino_vitbase8_pretrain.zip")
    backbone.load_state_dict(chk)
    backbone.to('cuda')
    # head = vits.DINOHead(backbone.embed_dim, 65536, False, True)
    # back = MultiCropWrapper(backbone, head)
    # out = back(torch.zeros((1,3,512,512)))
    with torch.no_grad():
        out = backbone(pil_to_tensor(Image.open("samples/23_image.png")).cuda().unsqueeze(0)/1.)
    to_pil_image(out.reshape(24,32)).show()

    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3plusedgev1':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3PlusEdge(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    elif name == 'ACE2P':
        return_layers = {'layer4': 'out', 'layer3': 'high_level', 'layer2': 'mid_level', 'layer1': 'low_level'}
        classifier = AugmentedCE2PHead(inplanes, low_level_planes, num_classes, aspp_dilate, use_abn=use_abn)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    if 'edgev1' in name and 'deeplabv3' in name:
        model = DeepLabV3Edge(backbone, classifier)
    elif 'deeplabv3' in name:
        model = DeepLabV3(backbone, classifier)
    elif name == 'ACE2P':
        model = ACE2P(backbone, classifier)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, ace2p=False, use_abn=False):
    if backbone == 'mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride,
                                pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone, ace2p=ace2p, use_abn=use_abn)
    elif backbone.startswith('vit'):
        model = _segm_vit(arch_type, backbone, num_classes, patch_size=output_stride,
                             pretrained_backbone=pretrained_backbone, ace2p=ace2p, use_abn=use_abn)
    else:
        raise NotImplementedError
    return model


# Deeplab v3

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn)


def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn)


def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn)


# Deeplab v3+

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False):
    """Constructs a DeepLabV3+ model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn)


def deeplabv3plus_resnet101_ver2(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101v2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn, ace2p=True)


def deeplabv3plus_vitb8(num_classes=21, output_stride=16, pretrained_backbone=True, use_abn=False):
    """Constructs a DeepLabV3+ model with a VisionTransform backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'vit_base', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn, ace2p=True)


def deeplabv3plusedgev1_resnet101_ver2(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plusedgev1', 'resnet101v2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn, ace2p=True)


# ACE2P

def ACE2P_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False):
    """Constructs a ACE2P model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('ACE2P', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn, ace2p=True)


def ACE2P_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False):
    """Constructs a ACE2P model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('ACE2P', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn, ace2p=True)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, use_abn=False):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, use_abn=use_abn)
