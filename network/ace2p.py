import torch
from torch import nn
from torch.nn import functional as F

from network.utils import _SimpleSegmentationModel


class ACE2P(nn.Module):
    def __init__(self, backbone, classifier):
        super(ACE2P, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x[0][0] = F.interpolate(x[0][0], size=input_shape, mode='bilinear', align_corners=False)
        x[0][1] = F.interpolate(x[0][1], size=input_shape, mode='bilinear', align_corners=False)
        x[1][0] = F.interpolate(x[1][0], size=input_shape, mode='bilinear', align_corners=False)
        return x



class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), use_abn=False):
        super(PSPModule, self).__init__()

        if use_abn:
            import functools
            from .modules.bn import InPlaceABNSync
            BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
        else:
            BatchNorm2d=nn.BatchNorm2d

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, BatchNorm2d) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            BatchNorm2d(out_features),
        )

    def _make_stage(self, features, out_features, size, BatchNorm2d):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

class Edge_Module(nn.Module):
    """
    Edge Learning Branch
    """

    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2, use_abn=False):
        super(Edge_Module, self).__init__()

        if use_abn:
            import functools
            from .modules.bn import InPlaceABNSync
            BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
        else:
            BatchNorm2d=nn.BatchNorm2d

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


class Decoder_Module(nn.Module):
    """
    Parsing Branch Decoder Module.
    """

    def __init__(self, num_classes, use_abn=False):
        super(Decoder_Module, self).__init__()

        if use_abn:
            import functools
            from .modules.bn import InPlaceABNSync
            BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
        else:
            BatchNorm2d=nn.BatchNorm2d

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(256)
        )

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x



class AugmentedCE2PHead(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36], use_abn=False):
        super(AugmentedCE2PHead, self).__init__()

        if use_abn:
            import functools
            from .modules.bn import InPlaceABNSync
            BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
        else:
            BatchNorm2d=nn.BatchNorm2d

        self.context_encoding = PSPModule(2048, 512, use_abn=use_abn)

        self.edge = Edge_Module(use_abn=use_abn)
        self.decoder = Decoder_Module(num_classes, use_abn=use_abn)

        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )
        if use_abn:
            import functools
            from .modules.bn import InPlaceABNSync
            BatchNorm2d = InPlaceABNSync
        else:
            BatchNorm2d=nn.BatchNorm2d
        self._init_weight(BatchNorm2d)

    def forward(self, feature):
        x2 = feature['low_level']
        x3 = feature['mid_level']
        x4 = feature['high_level']
        x5 = feature['out']
        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x, x2)
        # Edge Branch
        edge_result, edge_fea = self.edge(x2, x3, x4)
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x)
        return [[parsing_result, fusion_result], [edge_result]]

    def _init_weight(self, BatchNorm2d):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)