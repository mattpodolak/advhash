import torch
import numpy as np
from PIL import Image
from advhash.hash.dhash import DHash
from advhash.utils import compare
import imagehash

image = Image.open('../img/cat.jpg')
im_tensor = torch.tensor(np.array(image).astype('float32'))

def test_init():
    dhash = DHash(split_point="resize", hash_size=8)
    assert dhash.split_point == "resize" and dhash.hash_size == 8

def test_partial_hash():
    expected_resize = np.array(image.convert("L").resize((9, 8), Image.LANCZOS)).astype('float32')
    dhash = DHash(split_point="resize", hash_size=8)
    partial_hash = dhash.partial_hash(im_tensor)
    assert partial_hash.numpy().shape == expected_resize.shape

def test_last_hash():
    dhash = DHash(split_point="last", hash_size=8)
    partial_hash = dhash.partial_hash(im_tensor)
    assert partial_hash.numpy().shape == (8, 8)

def test_full_hash():
    expected_hash = np.array(imagehash.dhash(image).hash).astype('float32')
    dhash = DHash(split_point="resize", hash_size=8)
    full_hash = dhash.full_hash(im_tensor)
    dist = compare.hamming_dist(torch.tensor(expected_hash), full_hash)
    assert full_hash.numpy().shape == expected_hash.shape and dist.item() == 1 #TODO: should be 0

def test_init_device():
    device = torch.device('cpu')
    dhash = DHash(split_point="resize", hash_size=8, device=device)
    assert dhash.device == device

def test_init_str():
    dhash = DHash(split_point="resize", hash_size=8, device='cpu')
    assert dhash.device == torch.device('cpu')
