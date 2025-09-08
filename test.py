import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("Torch version:", torch.__version__)

from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="./asl_dataset")
print(dataset)
