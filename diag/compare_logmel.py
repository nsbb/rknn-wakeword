import torch
import torchaudio
import numpy as np
from inference_rknn import LogMel as NumpyLogMel
from inference import LogMel as TorchLogMel

import wave

# Generate dummy audio
dummy_audio = np.random.randn(16000).astype(np.float32)

numpy_logmel = NumpyLogMel(apply_preemph=False)
torch_logmel = TorchLogMel('cpu', apply_preemph=False)

# Torch transform
torch_tensor = torch.from_numpy(dummy_audio).unsqueeze(0)
torch_out = torch_logmel(torch_tensor)

# Numpy transform
numpy_out = numpy_logmel(dummy_audio)

print("Torch output shape:", torch_out.shape)
print("Torch output head:\n", "Min:", torch_out.min().item(), "Max:", torch_out.max().item(), "Mean:", torch_out.mean().item())

print("Numpy output shape:", numpy_out.shape)
print("Numpy output head:\n", "Min:", numpy_out.min(), "Max:", numpy_out.max(), "Mean:", numpy_out.mean())

diff = np.abs(torch_out.detach().numpy().squeeze() - numpy_out)
print("Mean Absolute Difference:", diff.mean())
print("Max Difference:", diff.max())

