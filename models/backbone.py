import  torchvision.models as pytorch_models
from models.baseline import BaseLineModel
from models.massive_attention import get_pretrained_net

backbone_list = {
  "resnet18": pytorch_models.resnet18,
  "resnet34": pytorch_models.resnet34,
  "resnet50": pytorch_models.resnet50,
  "wide_resnet50_2": pytorch_models.wide_resnet50_2,
  "massive_attention": get_pretrained_net
}

def get_back_bone(backbone, pretrained = True, num_classes = 2):
  if backbone == "massive_attention": return backbone_list[backbone](backbone, num_classes)

  backbone = backbone_list[backbone](pretrained)
  return BaseLineModel(backbone = backbone, keep = True)