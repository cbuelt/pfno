import torch

torch.manual_seed(12345)

for i in range(5):
    print('New state')
    newstate = torch.get_rng_state()
    for j in range(3):
        torch.set_rng_state(newstate)
        print(torch.randn(1).item())
