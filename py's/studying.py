import torch

new_tensor = torch.rand(10)
viewed_tensor = new_tensor.view(1, 2, 5)
print(viewed_tensor)
print(viewed_tensor.shape)
