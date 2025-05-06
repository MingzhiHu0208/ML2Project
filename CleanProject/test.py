import os, pathlib, glob
os.environ["XLA_FLAGS"] = r"--xla_gpu_cuda_data_dir=C:\Progra~1\NVIDIA~1\CUDA\v11.2\"
print("XLA_FLAGS =", os.getenv("XLA_FLAGS"))

root = pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2")
print("Exists:", root.exists())
print("libdevice files:", glob.glob(str(root / "nvvm/libdevice/libdevice*.bc")))