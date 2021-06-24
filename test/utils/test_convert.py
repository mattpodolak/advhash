import advhash.utils.convert as convert
import advhash.utils.compare as compare
from PIL import Image
import numpy as np
import torch

test_image = Image.open("../img/cat.jpg")

def test_rgb2luma_cuda():
    device = torch.device('cuda')
    test_tensor = torch.tensor(np.array(test_image, "float32")).to(device)
    test_resize = convert.rgb2luma(test_tensor, device)
    expected_resize = torch.tensor(np.array(test_image.convert("L"))).to(device)
    avg_diff = compare.avg_diff(expected_resize, test_resize)
    assert avg_diff < 0.005, f"Avg diff={100*avg_diff:.2f}% not less than 0.5%"

def test_rgb2luma_cpu():
    test_tensor = torch.tensor(np.array(test_image, "float32"))
    test_resize = convert.rgb2luma(test_tensor)
    expected_resize = torch.tensor(np.array(test_image.convert("L")))
    avg_diff = compare.avg_diff(expected_resize, test_resize)
    assert avg_diff < 0.005, f"Avg diff={100*avg_diff:.2f}% not less than 0.5%"

def test_tensor2image():
    test_tensor = torch.tensor(np.array(test_image, "float32"))
    image = convert.tensor2image(test_tensor)
    assert (np.array(image) == test_tensor.numpy()).all()

def test_image2tensor():
    arr = np.array([[0, 255, 0, 255]]).astype('float32')
    im = Image.fromarray(arr)
    tensor = convert.image2tensor(im)
    assert (arr == tensor.numpy()).all()
