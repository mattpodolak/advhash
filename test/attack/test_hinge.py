from advhash.utils import compare
from advhash.attack.hinge import HingeAttack
from advhash.hash.dhash import DHash
import torch
from PIL import Image
import numpy as np

image = Image.open('test/img/cat.jpg')
im_tensor = torch.tensor(np.array(image).astype('float32'))

def test_init_cpu():
    hinge = HingeAttack(hash_fn='dhash', device='cpu')
    partial_hash = hinge.hash.partial_hash(im_tensor)
    assert partial_hash.numpy().shape == (16, 16)

def test_init_partial():
    dhash = DHash(split_point='last', device='cpu')
    hinge = HingeAttack(hash_fn=dhash)
    partial_hash = hinge.hash.partial_hash(im_tensor)
    assert partial_hash.cpu().numpy().shape == (16, 16)    

def test_attack_cpu():
    hinge = HingeAttack(hash_fn='dhash', device='cpu')
    target = torch.tensor(np.array(Image.open('test/img/forest.jpg')).astype('float32'))
    target_hash = hinge.hash.full_hash(target)
    im_hash = hinge.hash.full_hash(im_tensor)
    im_adv = hinge.attack(im_tensor, target, 2)
    
    adv_hash = hinge.hash.full_hash(im_adv)
    assert compare.hamming_dist(adv_hash, target_hash) < compare.hamming_dist(im_hash, target_hash)
