import network

model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
}