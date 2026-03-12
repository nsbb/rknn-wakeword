import numpy as np

n_fft = 512
win_length = 480
hop_length = 160

# 1.5초 * 16000
audio_len = int(1.5 * 16000)
print(f"Audio len: {audio_len}")

# Pad
pad_size = n_fft // 2
padded_audio_len = audio_len + 2 * pad_size
print(f"Padded len: {padded_audio_len}")

# frames 계산: (padded_len - win_length) // hop_length + 1
# 참고: torchaudio는 (padded_len - n_fft) // hop_length + 1 인 경우도 있음
frames = (padded_audio_len - n_fft) // hop_length + 1
print(f"Frames calculation 1: {frames}")

