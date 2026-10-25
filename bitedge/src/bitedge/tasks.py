import torch


def teacher_student(n_train=20000, n_test=4000, d_in=64, seed=1234, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_train + n_test, d_in, generator=g)
    h = X
    for k in range(4):
        W = torch.randn(d_in if k == 0 else 48, 48, generator=g) / (8 if k == 0 else 7)
        h = torch.tanh(h @ W)
    y = (h @ torch.randn(48, 10, generator=g)).argmax(1)
    X, y = X.to(device), y.to(device)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]
