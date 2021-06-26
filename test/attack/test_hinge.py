from advhash.utils import compare
from advhash.attack.hinge import HingeAttack
from advhash.hash.dhash import DHash
import torch
from PIL import Image
import numpy as np

image = Image.open('../img/cat.jpg')
im_tensor = torch.tensor(np.array(image).astype('float32'))

def test_init_cpu():
    hinge = HingeAttack(hash_fn='dhash', device='cpu')
    partial_hash = hinge.hash.partial_hash(im_tensor)
    assert partial_hash.numpy().shape == (16, 16)

def test_init_cuda():
    hinge = HingeAttack(hash_fn='dhash', device='cuda')
    partial_hash = hinge.hash.partial_hash(im_tensor)
    assert partial_hash.cpu().numpy().shape == (16, 16)

def test_init_partial():
    dhash = DHash(split_point='last', device='cuda')
    hinge = HingeAttack(hash_fn=dhash)
    partial_hash = hinge.hash.partial_hash(im_tensor)
    assert partial_hash.cpu().numpy().shape == (16, 16)    

def test_attack_cuda():
    hinge = HingeAttack(hash_fn='dhash', device='cuda')
    target = torch.tensor(np.array(Image.open('../img/forest.jpg')).astype('float32'))
    target_hash = hinge.hash.full_hash(target).to('cuda')
    im_hash = hinge.hash.full_hash(im_tensor).to('cuda')
    im_adv = hinge.attack(im_tensor, target, 2)
    
    adv_hash = hinge.hash.full_hash(im_adv)
    assert compare.hamming_dist(adv_hash, target_hash) < compare.hamming_dist(im_hash, target_hash)
