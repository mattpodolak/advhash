import advhash.utils.resize as resize
import advhash.utils.compare as compare
from PIL import Image
import numpy as np
import torch

test_image = Image.open("../img/cat.jpg").convert("L")

def test_lanczos_resize_cuda():
    device = torch.device('cuda')
    test_tensor = torch.tensor(np.array(test_image, "float32")).to(device)
    test_resize = resize.lanczos_resize(test_tensor)
    expected_resize = torch.tensor(np.array(test_image.resize((17, 16), Image.ANTIALIAS))).to(device)
    avg_diff = compare.avg_diff(expected_resize, test_resize)
    assert avg_diff < 0.005, f"Avg diff={100*avg_diff:.2f}% not less than 0.5%"

def test_lanczos_resize_cpu():
    test_tensor = torch.tensor(np.array(test_image, "float32"))
    test_resize = resize.lanczos_resize(test_tensor)
    expected_resize = torch.tensor(np.array(test_image.resize((17, 16), Image.ANTIALIAS)))
    avg_diff = compare.avg_diff(expected_resize, test_resize)
    assert avg_diff < 0.005, f"Avg diff={100*avg_diff:.2f}% not less than 0.5%"

def test_sinc():
    assert resize.sinc(0) == torch.tensor(1.0), "sinc(0) should return 1.0"

def test_lanczos3():
    assert resize.lanczos3(3) == torch.tensor(0.0), "lanczos3(3) should return 0.0"
