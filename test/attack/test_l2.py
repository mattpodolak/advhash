from advhash.attack.l2 import L2Attack

def test_init_cpu():
    l2 = L2Attack(hash_fn='dhash', split_point='resize', device='cpu')
    assert isinstance(l2, L2Attack)
