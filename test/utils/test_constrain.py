import advhash.utils.constrain as constrain
from PIL import Image
import numpy as np
import torch

test_image = Image.open("../img/cat.jpg")

def test_box():
  test_tensor = torch.tensor(np.array(test_image, "float32")).cuda()
  test_box = constrain.box(test_tensor*3)
  assert torch.max(test_box) <= 255 and torch.min(test_box) >= 0, "Values should be constrained between 0 and 255"

def test_box_conv():
  test_tensor = torch.tensor(np.array(test_image, "float32")).cuda()
  test_box_conv = constrain.box_conv(test_tensor)
  assert torch.max(test_box_conv) == torch.abs(torch.min(test_box_conv)), "Values should be distributed between max==min"