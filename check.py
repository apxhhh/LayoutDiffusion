import torch

import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device Count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i} Name:", torch.cuda.get_device_name(i))
else:
    print("CUDA is not available.")
