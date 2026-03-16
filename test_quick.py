"""단일 WAV 파일로 웨이크워드 감지 테스트.

사용법:
    conda run -n RKNN-Toolkit2 python test_quick.py <wav_file>
    conda run -n RKNN-Toolkit2 python test_quick.py  # 기본 테스트 파일 사용
"""
import sys, numpy as np, wave, time, os
from inference_rknn import LogMel
from rknnlite.api import RKNNLite

MODEL = 'models/BCResNet-t2-npu-fixed.rknn'
THRESHOLD = 0.55

def load_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
    if sr != 16000:
        old_idx = np.linspace(0, len(audio)/sr, len(audio))
        new_idx = np.linspace(0, len(audio)/sr, int(len(audio) * 16000 / sr))
        audio = np.interp(new_idx, old_idx, audio)
    return audio

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

if __name__ == '__main__':
    wav_path = sys.argv[1] if len(sys.argv) > 1 else 'wallpad_HiWonder_251113/lkk/lkk_1_2.wav'
    if not os.path.exists(wav_path):
        print(f"파일 없음: {wav_path}")
        sys.exit(1)

    audio = load_wav(wav_path)
    # 1.5초(24000 samples)로 맞추기
    if len(audio) > 24000:
        audio = audio[:24000]
    elif len(audio) < 24000:
        audio = np.pad(audio, (0, 24000 - len(audio)))

    logmel = LogMel()
    feat = logmel(audio)[np.newaxis, np.newaxis, ...]

    rknn = RKNNLite(verbose=False)
    rknn.load_rknn(MODEL)
    rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    # warmup
    rknn.inference(inputs=[feat], data_format='nchw')

    t0 = time.perf_counter()
    raw = rknn.inference(inputs=[feat], data_format='nchw')[0]
    elapsed = (time.perf_counter() - t0) * 1000

    probs = softmax(raw.squeeze())
    wake_prob = float(probs[1])
    detected = wake_prob >= THRESHOLD

    print(f"파일: {wav_path}")
    print(f"웨이크워드 확률: {wake_prob:.4f} (threshold: {THRESHOLD})")
    print(f"결과: {'감지됨' if detected else '감지 안됨'}")
    print(f"NPU 추론: {elapsed:.1f}ms")

    rknn.release()
