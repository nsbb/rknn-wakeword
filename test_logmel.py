import numpy as np

def compute_mel_spectrogram(waveform, n_fft=512, win_length=480, hop_length=160):
    pad_size = n_fft // 2
    waveform = np.pad(waveform, (pad_size, pad_size), mode='reflect')
    
    frames = []
    # torchaudio center=True default uses hop_length
    # The number of frames is usually 1 + int(len(waveform_not_padded) / hop_length) = 1 + 24000/160 = 151
    # Let's adjust the loop
    for i in range(0, len(waveform) - n_fft + 1, hop_length):
        chunk = waveform[i:i+n_fft]
        frames.append(chunk)

    print(f"Number of frames: {len(frames)}")
    return frames

waveform = np.random.randn(int(1.5 * 16000)).astype(np.float32)
compute_mel_spectrogram(waveform)
