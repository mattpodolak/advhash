from PIL import Image
import numpy as np
import torch
import advhash.utils.compare as compare

test_image = Image.open("../img/cat.jpg").convert("L")

def test_avg_diff():
    test_tensor = torch.tensor(np.array(test_image, "float32"))
    mod_tensor = test_tensor*0.9
    avg_diff = compare.avg_diff(test_tensor, mod_tensor)
    assert round(avg_diff.item(), 7) == 0.1, f"Expected a difference of 0.1000000, \
        instead got {avg_diff:.7f}"

def test_hamming_dist():
    ones = torch.ones((5, 5))
    zeros = torch.zeros((5, 5))
    dist = compare.hamming_dist(ones, zeros)
    assert dist.item() == 25, f"Expected a distance of 25, instead got {dist}"
