import torch

def generate_mcd_samples(model, x, shape, n_samples=100):
    samples = torch.zeros(*shape, n_samples).to(x.device)
    for i in range(n_samples):
        with torch.no_grad():
            samples[...,i] = model(x).detach()
    return samples