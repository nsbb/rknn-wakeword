import torch
import torchaudio

# Let's generate dummy data and check Torchaudio's exact output shape and values
# Since torch is missing in RKNN-Toolkit2 env, try base env

# If this script runs in an environment with torch installed
dummy_audio = torch.randn(1, 24000)
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    hop_length=160,
    n_fft=512,
    win_length=480,
    n_mels=40,
)
spec = mel(dummy_audio)

# Torchaudio default power is 2.0. So it is a power spectrogram.
# Usually, people apply torchaudio.transforms.AmplitudeToDB() to get LogMel.
db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
log_spec = db_transform(spec)

print("Spec shape:", spec.shape)
print("LogSpec shape:", log_spec.shape)
print("LogSpec Mean:", log_spec.mean().item())

