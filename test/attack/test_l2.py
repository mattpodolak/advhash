from advhash.utils import compare
from advhash.attack.l2 import L2Attack
from advhash.hash.dhash import DHash
import torch
from PIL import Image
import numpy as np

image = Image.open('../img/cat.jpg')
im_tensor = torch.tensor(np.array(image).astype('float32'))
expected_resize = np.array(image.convert("L").resize((17, 16), Image.LANCZOS)).astype('float32')

def test_init_cpu():
    l2 = L2Attack(hash_fn='dhash', split_point='resize', device='cpu')
    partial_hash = l2.hash.partial_hash(im_tensor)
    assert partial_hash.numpy().shape == expected_resize.shape

def test_init_cuda():
    l2 = L2Attack(hash_fn='dhash', split_point='resize', device='cuda')
    partial_hash = l2.hash.partial_hash(im_tensor)
    assert partial_hash.cpu().numpy().shape == expected_resize.shape

def test_init_partial():
    dhash = DHash(split_point='resize', device='cuda')
    l2 = L2Attack(hash_fn=dhash)
    partial_hash = l2.hash.partial_hash(im_tensor)
    assert partial_hash.cpu().numpy().shape == expected_resize.shape    

def test_attack_cuda():
    l2 = L2Attack(hash_fn='dhash', split_point='resize', device='cuda')
    target = torch.tensor(np.array(Image.open('../img/forest.jpg')).astype('float32'))
    target_hash = l2.hash.full_hash(target).to('cuda')
    im_hash = l2.hash.full_hash(im_tensor).to('cuda')
    im_adv = l2.attack(im_tensor, target, 2)
    
    adv_hash = l2.hash.full_hash(im_adv)
    assert compare.hamming_dist(adv_hash, target_hash) < compare.hamming_dist(im_hash, target_hash)
