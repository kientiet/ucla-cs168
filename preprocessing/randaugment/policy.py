from preprocessing.randaugment.randaugmentation import ALL_TRANSFORMS
import numpy as np

def get_trans_list():
  trans_list = [
      'Invert', 'Cutout', 'Sharpness', 'AutoContrast', 'Posterize',
      'ShearX', 'TranslateX', 'TranslateY', 'ShearY', 'Rotate',
      'Equalize', 'Contrast', 'Color', 'Solarize', 'Brightness']
  return trans_list

def randaug_policies():
  trans_list = get_trans_list()
  op_list = []
  for trans in trans_list:
    for magnitude in range(1, 10):
      op_list += [(trans, 0.5, magnitude)]

  policies = []
  for op_1 in op_list:
    for op_2 in op_list:
      policies += [[op_1, op_2]]

  return np.array(policies)