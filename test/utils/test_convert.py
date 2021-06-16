import advhash.utils.convert as convert
import advhash.utils.compare as compare
from PIL import Image
import numpy as np
import torch

test_image = Image.open("../img/cat.jpg")

def test_rgb2luma():
  test_tensor = torch.tensor(np.array(test_image, "float32")).cuda()
  test_resize = convert.rgb2luma(test_tensor)
  expected_resize = torch.tensor(np.array(test_image.convert("L"))).cuda()
  avg_diff = compare.avg_diff(expected_resize, test_resize)
  assert avg_diff < 0.005, f"Avg diff={100*avg_diff:.2f}% not less than 0.5%"