from PIL import Image
import numpy as np
import torch
import advhash.utils.constrain as constrain

test_image = Image.open("../img/cat.jpg")

def test_box():
    test_tensor = torch.tensor(np.array(test_image, "float32")).cuda()
    box = constrain.box(test_tensor*3)
    assert torch.max(box) <= 255 and torch.min(box) >= 0, \
        "Values should be constrained between 0 and 255"

def test_box_conv():
    test_tensor = torch.tensor(np.array(test_image, "float32")).cuda()
    box = constrain.box_conv(test_tensor)
    assert torch.max(box) == torch.abs(torch.min(box)), \
        "Values should be distributed between max==min"
