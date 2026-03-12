import numpy as np
# Let's inspect the np.log10 implementation
def apply_log_transform(mel_spec: np.ndarray) -> np.ndarray:
    return np.log10(mel_spec + 1e-6)

dummy_mel = np.array([0.0, 1e-10, 1e-7, 1e-6, 1e-5, 1.0])
print(apply_log_transform(dummy_mel))
