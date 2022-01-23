import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# EfficientNet-b4
class EffiNet(nn.Module):
    def __init__(self,classes):
        super(EffiNet, self).__init__()
        self.classes=classes
        self.base_model = EfficientNet.from_pretrained("efficientnet-b4")
        num_ftrs = self.base_model._fc.in_features
        self.base_model._fc = nn.Linear(num_ftrs,self.classes, bias = True)
        
    def forward(self, image):
        out = self.base_model(image)
        return out

# Resnet50
import torchvision
class Resnet50(nn.Module):
    def __init__(self,classes):
        super(Resnet50, self).__init__()
        self.classes=classes
        self.base_model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs,self.classes, bias = True)
        
    def forward(self, image):
        out = self.base_model(image)
        return out       
# densenet
import torchvision
class Densenet(nn.Module):
    def __init__(self,classes):
        super(Densenet, self).__init__()
        self.classes=classes
        self.base_model = torchvision.models.densenet201(pretrained=True)
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(num_ftrs,self.classes, bias = True)
        
    def forward(self, image):
        out = self.base_model(image)
        return out    
# EfficientNet-b7
class EffiNet7(nn.Module):
    def __init__(self,classes):
        super(EffiNet7, self).__init__()
        self.classes=classes
        self.base_model = EfficientNet.from_pretrained("efficientnet-b7")
        num_ftrs = self.base_model._fc.in_features
        self.base_model._fc = nn.Linear(num_ftrs,self.classes, bias = True)
        
    def forward(self, image):
        out = self.base_model(image)
        return out  