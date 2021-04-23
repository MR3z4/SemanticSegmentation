import network

model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'ACE2P_resnet50': network.ACE2P_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3plus_resnet101v2': network.deeplabv3plus_resnet101_ver2,
    'deeplabv3plusedge_resnet101v2': network.deeplabv3plusedge_resnet101_ver2,
    'ACE2P_resnet101': network.ACE2P_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
}