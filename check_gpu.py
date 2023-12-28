import torch
x = torch.rand(5, 3)
print(f"x = {x}")

print(f"check gpu: {torch.cuda.is_available()}")