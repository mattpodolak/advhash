from advhash.hash.dhash import DHash

dhash = DHash(split_point="resize", hash_size=8)

def test_init():
    assert dhash.split_point == "resize" and dhash.hash_size == 8

# TODO: complete hash fn testing
# def test_partial_hash():
#     assert dhash.partial_hash() == 

# def test_full_hash():
#     assert dhash.full_hash()