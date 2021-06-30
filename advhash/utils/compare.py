import torch

def avg_diff(X_orig, X_pert):
    return torch.mean(torch.abs(X_pert - X_orig)/X_orig)

def hamming_dist(X_hash, Y_hash):
    return torch.sum(torch.abs(X_hash - Y_hash))
