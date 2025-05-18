import torch, torchvision, torchaudio
print("torch", torch.__version__)
print("cuda ", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
print("compiler API?", hasattr(torch, "compiler"))
print("torchvision", torchvision.__version__)
print("torchaudio",  torchaudio.__version__)
