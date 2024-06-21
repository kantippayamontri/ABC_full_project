import torch
x = torch.rand(5, 3)
print(f"x = {x}")

if torch.cuda.is_available():
    print(f"check gpu: {torch.cuda.is_available()}")
    print(f"number of gpu: {torch.cuda.device_count()}")
else:
    print("not found gpu, use cpu instead.")