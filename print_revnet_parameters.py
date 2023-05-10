import torch
import resnet
import revnet

import torchvision

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Prints a bunch of revnet and resnet models to compare parameter counts.
# I just want to pick a revnet configuration that has roughly the same #parameters as resnet50.
if __name__ == '__main__':
    print("Printing RevNets and parameter counts:")    
    revnets = {
        "revnet-bottleneck-5-5-5-5" : revnet.RevNet(
            units=[5, 5, 5, 5],
            filters=[64, 64, 128, 256, 512],
            strides=[1, 1, 1, 1],
            classes=100,
            bottleneck=True
            ),
        "revnet3-3-3" : revnet.revnet38(),
        "revnet3-3-3-4" : revnet.RevNet(
            units=[3, 3, 3, 4],
            filters=[64, 64, 128, 256, 512],
            strides=[1, 1, 1, 1],
            classes=100,
        ),
    }
    for name in revnets:
        print(f"Model: {name} has number of parameters: {count_parameters(revnets[name])}")

    print("Printing (Adjusted for filters=[32, 32, 64, 112]) ResNets and parameter counts:")    
    print("TBC")
    

    print("Printing (Assignment 2) ResNets and parameter counts:")    
    resnets = {
        "resnet18" : resnet.ResNet18(),
        "resnet34" : resnet.ResNet34(),
        "resnet50" : resnet.ResNet50()
    }
    for name in resnets:
        print(f"Model: {name} has number of parameters: {count_parameters(resnets[name])}")

    print("Printing pytorch ResNets and parameter counts:")   
    torch_resnets = {
        "resnet18" : torchvision.models.resnet18(),
        "resnet50": torchvision.models.resnet50()
    }
    for name in torch_resnets:
        print(f"Model: {name} has number of parameters: {count_parameters(torch_resnets[name])}")