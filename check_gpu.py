import torch
import os
x = torch.rand(5, 3)
print(f"x = {x}")

if torch.cuda.is_available():
    print(f"check gpu: {torch.cuda.is_available()}")
    print(f"number of gpu: {torch.cuda.device_count()}")
    print(f"number of cpu cores : {os.cpu_count()}")
else:
    print("not found gpu, use cpu instead.")
    print(f"number of cpu cores : {os.cpu_count()}")

# import torch
# import torchvision.models as models

# # Check if GPU is available
# if not torch.cuda.is_available():
#     print("GPU not available.")
#     exit()

# # Get GPU properties
# gpu_properties = torch.cuda.get_device_properties(0)
# total_memory = gpu_properties.total_memory
# print(f"Total GPU memory: {total_memory / (1024 ** 2):.2f} MB")

# # Load a model (example: ResNet18)
# model = models.resnet18().cuda()
# model.eval()

# # Create dummy input data
# batch_size = 1
# image_size = (3, 640, 640)  # Example image size (3 channels, 224x224)

# def get_memory_usage():
#     return torch.cuda.memory_allocated(0) / (1024 ** 2)  # Convert bytes to MB

# # Estimate maximum batch size
# while True:
#     try:
#         dummy_input = torch.randn(batch_size, *image_size).cuda()
#         with torch.no_grad():
#             output = model(dummy_input)
#         memory_usage = get_memory_usage()
#         print(f"Batch size: {batch_size}, Memory usage: {memory_usage:.2f} MB")
#         batch_size += 1
#     except RuntimeError as e:
#         print(f"Out of memory at batch size: {batch_size}")
#         break