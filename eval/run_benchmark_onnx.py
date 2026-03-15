"""
BCResNet-t2 ONNX 추론 성능 및 정확도 측정 스크립트
- 속도: ONNX only (N=200, warmup=10) / LogMel+ONNX (N=100)
- 정확도: test.csv로 Micro/Macro/Per-class Accuracy + Confusion Matrix
- FAR: vad_cropped/ 에서 4가지 설정별 False Alarm Rate/hour
"""

import os
import sys
import gc
import time
import wave
import glob
import random
import datetime
import numpy as np
import pandas as pd
from collections import deque
from tqdm.auto import tqdm

# ── ONNX Runtime ─────────────────────────────────────────────
import onnxruntime as ort

ORT_VERSION = ort.__version__
ONNX_PATH = "../models/BCResNet-t2-Focal-ep110.onnx"
TEST_CSV   = "../test.csv"
VAD_DIR    = "../vad_cropped"
BASE_DIR   = ".."
OUT_MD     = "../benchmark_results.md"

# ONNX 세션 생성
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
session = ort.InferenceSession(ONNX_PATH, sess_options)
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 모델 파일 크기
model_size_kb = os.path.getsize(ONNX_PATH) // 1024

# ── 전처리 클래스 (inference_rknn.py 와 동일) ────────────────

class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load_audio(self, audio_path: str):
        with wave.open(audio_path, 'rb') as wf:
            sr         = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            n_frames   = wf.getnframes()
            data       = wf.readframes(n_frames)
            dtype      = np.int16 if sampwidth == 2 else np.uint8
            audio      = np.frombuffer(data, dtype=dtype)
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128.0) / 128.0
        return audio, sr

    def convert_to_mono(self, w: np.ndarray) -> np.ndarray:
        return np.mean(w, axis=1) if len(w.shape) > 1 else w

    def resample(self, w: np.ndarray, source_sr: int) -> np.ndarray:
        if source_sr != self.target_sr:
            duration    = len(w) / source_sr
            num_samples = int(duration * self.target_sr)
            old_idx = np.linspace(0, duration, len(w))
            new_idx = np.linspace(0, duration, num_samples)
            w = np.interp(new_idx, old_idx, w)
        return w

    def pad_or_truncate(self, w: np.ndarray, target_length: int = 24000) -> np.ndarray:
        if len(w) > target_length:
            return w[:target_length]
        elif len(w) < target_length:
            return np.pad(w, (0, target_length - len(w)), mode='constant')
        return w

    def load_and_preprocess(self, audio_path: str) -> np.ndarray:
        w, sr = self.load_audio(audio_path)
        w     = self.convert_to_mono(w)
        w     = self.resample(w, sr)
        return w


class LogMel:
    def __init__(self, sample_rate=16000, hop_length=160, win_length=480,
                 n_fft=512, n_mels=40, apply_preemph=False):
        self.sr           = sample_rate
        self.hop_length   = hop_length
        self.win_length   = win_length
        self.n_fft        = n_fft
        self.n_mels       = n_mels
        self.apply_preemph = apply_preemph
        self.mel_basis    = self._create_mel_filterbank()
        self.window       = np.hanning(win_length)

    def _create_mel_filterbank(self):
        def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700.0)
        def mel_to_hz(mel): return 700 * (10**(mel / 2595.0) - 1)
        all_freqs  = np.linspace(0, self.sr / 2, self.n_fft // 2 + 1)
        mel_points = np.linspace(hz_to_mel(0), hz_to_mel(self.sr / 2), self.n_mels + 2)
        hz_points  = mel_to_hz(mel_points)
        fb = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            left, center, right = hz_points[i:i+3]
            for j, f in enumerate(all_freqs):
                if left < f < center:
                    fb[i, j] = (f - left) / (center - left)
                elif center <= f < right:
                    fb[i, j] = (right - f) / (right - center)
        return fb

    def apply_preemphasis(self, w: np.ndarray) -> np.ndarray:
        return np.append(w[0], w[1:] - 0.97 * w[:-1])

    def compute_mel_spectrogram(self, w: np.ndarray) -> np.ndarray:
        pad_size = self.n_fft // 2
        w        = np.pad(w, (pad_size, pad_size), mode='reflect')
        frames   = []
        for i in range(0, len(w) - self.n_fft + 1, self.hop_length):
            chunk = w[i:i+self.win_length]
            if len(chunk) < self.n_fft:
                chunk = np.pad(chunk, (0, self.n_fft - len(chunk)), mode='constant')
            chunk = chunk * np.pad(self.window, (0, self.n_fft - len(self.window)), mode='constant')
            spec  = np.abs(np.fft.rfft(chunk, n=self.n_fft))**2
            frames.append(spec)
        spectrogram = np.array(frames).T
        mel_spec    = np.dot(self.mel_basis, spectrogram)
        return mel_spec

    def apply_log_transform(self, mel_spec: np.ndarray) -> np.ndarray:
        return np.log(mel_spec + 1e-6)

    def __call__(self, w: np.ndarray) -> np.ndarray:
        if self.apply_preemph:
            w = self.apply_preemphasis(w)
        target_length = 24000
        if len(w) > target_length:
            w = w[:target_length]
        elif len(w) < target_length:
            w = np.pad(w, (0, target_length - len(w)), mode='constant')
        mel_spec = self.compute_mel_spectrogram(w)
        log_mel  = self.apply_log_transform(mel_spec)
        return log_mel.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def onnx_infer(feat: np.ndarray) -> np.ndarray:
    out = session.run([output_name], {input_name: feat})[0]
    return softmax(out.squeeze())


# ═══════════════════════════════════════════════════════════════
# 1. 속도 측정
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. 속도 측정")
print("="*60)

preprocessor = AudioPreprocessor()
logmel       = LogMel(apply_preemph=False)

# 샘플 오디오 파일 하나 준비
sample_wav = None
for candidate in glob.glob(os.path.join(BASE_DIR, "wallpad_HiWonder_251113", "**", "*.wav"), recursive=True):
    sample_wav = candidate
    break
if sample_wav is None:
    for candidate in glob.glob(os.path.join(VAD_DIR, "*.wav")):
        sample_wav = candidate
        break

print(f"Sample WAV: {sample_wav}")
raw_audio, raw_sr = preprocessor.load_audio(sample_wav)
raw_audio = preprocessor.convert_to_mono(raw_audio)
raw_audio = preprocessor.resample(raw_audio, raw_sr)
raw_audio = preprocessor.pad_or_truncate(raw_audio)

# 미리 feature 준비 (ONNX only 측정용)
feat_fixed = logmel(raw_audio)[np.newaxis, np.newaxis, ...]

# warmup (10회)
WARMUP = 10
for _ in range(WARMUP):
    session.run([output_name], {input_name: feat_fixed})

# (a) ONNX only: N=200
N_ONNX = 200
times_onnx = []
for _ in range(N_ONNX):
    t0 = time.perf_counter()
    session.run([output_name], {input_name: feat_fixed})
    t1 = time.perf_counter()
    times_onnx.append((t1 - t0) * 1000)

onnx_mean = np.mean(times_onnx)
onnx_min  = np.min(times_onnx)
onnx_max  = np.max(times_onnx)
onnx_std  = np.std(times_onnx)
print(f"ONNX only   → mean={onnx_mean:.2f}ms  min={onnx_min:.2f}ms  max={onnx_max:.2f}ms  std={onnx_std:.2f}ms")

# (b) LogMel + ONNX: N=100
N_TOTAL = 100
times_total = []
for _ in range(N_TOTAL):
    t0 = time.perf_counter()
    f  = logmel(raw_audio)[np.newaxis, np.newaxis, ...]
    session.run([output_name], {input_name: f})
    t1 = time.perf_counter()
    times_total.append((t1 - t0) * 1000)

total_mean = np.mean(times_total)
total_min  = np.min(times_total)
total_max  = np.max(times_total)
total_std  = np.std(times_total)
print(f"LogMel+ONNX → mean={total_mean:.2f}ms  min={total_min:.2f}ms  max={total_max:.2f}ms  std={total_std:.2f}ms")

logmel_mean = total_mean - onnx_mean
print(f"LogMel only (추정) → {logmel_mean:.2f}ms")

# ═══════════════════════════════════════════════════════════════
# 2. 정확도 측정 (test.csv)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. 정확도 측정 (test.csv)")
print("="*60)

test_df = pd.read_csv(TEST_CSV)
n_total_samples = len(test_df)
print(f"Total samples: {n_total_samples}")

y_true, y_pred = [], []
errors = 0

for _, row in tqdm(test_df.iterrows(), total=n_total_samples, desc="Accuracy Eval"):
    path  = row['path']
    label = int(row['label'])
    # 상대경로면 BASE_DIR 기준으로 변환
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)

    try:
        audio = preprocessor.load_and_preprocess(path)
        audio = preprocessor.pad_or_truncate(audio)
        feat  = logmel(audio)[np.newaxis, np.newaxis, ...]
        probs = onnx_infer(feat)
        pred  = int(np.argmax(probs))
        y_true.append(label)
        y_pred.append(pred)
    except Exception as e:
        errors += 1

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion Matrix
classes = sorted(set(y_true))
n_cls   = len(classes)
cm      = np.zeros((n_cls, n_cls), dtype=int)
for yt, yp in zip(y_true, y_pred):
    cm[yt][yp] += 1

per_class_acc = cm.diagonal() / cm.sum(axis=1)
micro_acc     = np.mean(y_true == y_pred)
macro_acc     = np.mean(per_class_acc)

print(f"Micro Accuracy : {micro_acc*100:.2f}%")
print(f"Macro Accuracy : {macro_acc*100:.2f}%")
for c in classes:
    print(f"  Class {c} Accuracy: {per_class_acc[c]*100:.2f}%  ({cm[c,c]}/{cm[c].sum()})")
print(f"Confusion Matrix:\n{cm}")
if errors:
    print(f"  (오류 건수: {errors})")

# ═══════════════════════════════════════════════════════════════
# 3. FAR 평가 (vad_cropped/)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. FAR 평가 (vad_cropped/)")
print("="*60)

WAV_FILES = sorted(glob.glob(os.path.join(VAD_DIR, "*.wav")))
print(f"Found {len(WAV_FILES)} WAV files in vad_cropped/")

# 최소 10분 분량 샘플링 (크면 랜덤 샘플 사용)
# 각 파일 길이 추정: 약 111MB / 1139 files ≈ 약 100KB/file
# WAV 16kHz 16bit mono → 32KB/s → 약 3초/파일
# 전체 약 57분. 전체 사용.
# 만약 RAM/시간이 부족하면 최대 200파일로 제한
MAX_FILES = 200
if len(WAV_FILES) > MAX_FILES:
    random.seed(42)
    WAV_FILES_SELECTED = random.sample(WAV_FILES, MAX_FILES)
    print(f"  → {MAX_FILES}개 파일로 샘플링 (seed=42)")
else:
    WAV_FILES_SELECTED = WAV_FILES
    print(f"  → 전체 {len(WAV_FILES_SELECTED)}개 파일 사용")

EMA_ALPHA  = 0.3
N_OF_M_N   = 3
N_OF_M_M   = 5
REFRAC_SEC = 2.0
THRESHOLD  = 0.5
WIN_SEC    = 1.5
SHIFT_SEC  = 0.2

CONFIGS = [
    {"name": "Raw (no processing)",          "REFRACTORY_SEC": 0.0,       "USE_EMA": False, "USE_N_OF_M": False},
    {"name": "Refractory only",              "REFRACTORY_SEC": REFRAC_SEC, "USE_EMA": False, "USE_N_OF_M": False},
    {"name": "Refractory + EMA",             "REFRACTORY_SEC": REFRAC_SEC, "USE_EMA": True,  "USE_N_OF_M": False},
    {"name": "Refractory + EMA + N-of-M",   "REFRACTORY_SEC": REFRAC_SEC, "USE_EMA": True,  "USE_N_OF_M": True},
]

total_results = {i: {"false_alarms": 0, "duration": 0.0} for i in range(len(CONFIGS))}
win_samples  = int(WIN_SEC  * 16000)
hop_samples  = int(SHIFT_SEC * 16000)

for wav_path in tqdm(WAV_FILES_SELECTED, desc="FAR Eval"):
    try:
        audio_data, sr = preprocessor.load_audio(wav_path)
        audio_data = preprocessor.convert_to_mono(audio_data)
        audio_data = preprocessor.resample(audio_data, sr)
        file_dur   = len(audio_data) / 16000.0

        # 슬라이딩 윈도우로 추론 (1회)
        all_probs = []
        all_times = []
        start = 0
        while start < len(audio_data):
            end   = start + win_samples
            if end > len(audio_data):
                chunk = np.pad(audio_data[start:], (0, end - len(audio_data)), mode='constant')
            else:
                chunk = audio_data[start:end]
            feat  = logmel(chunk)[np.newaxis, np.newaxis, ...]
            probs = onnx_infer(feat)
            all_probs.append(float(probs[1]))
            all_times.append(start / 16000.0)
            start += hop_samples

        all_probs = np.array(all_probs)

        # 설정별 트리거 카운트
        for cfg_idx, cfg in enumerate(CONFIGS):
            p_ema      = None
            trigbuf    = deque(maxlen=N_OF_M_M)
            cooldown   = -1e9
            fa_count   = 0

            for i, (p, t_start) in enumerate(zip(all_probs, all_times)):
                # EMA 업데이트
                if cfg["USE_EMA"]:
                    p_ema = EMA_ALPHA * p + (1 - EMA_ALPHA) * p_ema if p_ema is not None else p
                    p_s   = p_ema
                else:
                    p_s   = p

                hit = p_s >= THRESHOLD

                if cfg["USE_N_OF_M"]:
                    trigbuf.append(1 if hit else 0)
                    fired = (len(trigbuf) == N_OF_M_M) and (sum(trigbuf) >= N_OF_M_N)
                else:
                    fired = hit

                if t_start >= cooldown and fired:
                    fa_count += 1
                    cooldown  = t_start + cfg["REFRACTORY_SEC"]
                    p_ema     = None
                    trigbuf   = deque(maxlen=N_OF_M_M)

            total_results[cfg_idx]["false_alarms"] += fa_count
            total_results[cfg_idx]["duration"]     += file_dur

        del audio_data, all_probs
        gc.collect()

    except Exception as e:
        print(f"\nError: {wav_path}: {e}")

# FAR 요약 출력
print(f"\n{'='*60}")
print("FAR Summary (per hour):")
print(f"{'='*60}")
far_rows = []
for cfg_idx, cfg in enumerate(CONFIGS):
    fa   = total_results[cfg_idx]["false_alarms"]
    dur  = total_results[cfg_idx]["duration"]
    fph  = (fa / dur) * 3600.0 if dur > 0 else 0.0
    print(f"  [{cfg_idx+1}] {cfg['name']}")
    print(f"      FA={fa}, Duration={dur/60:.1f}min, FAR={fph:.2f}/hr")
    far_rows.append({
        "name": cfg["name"],
        "false_alarms": fa,
        "duration_min": dur / 60.0,
        "far_per_hour": fph,
    })

# ═══════════════════════════════════════════════════════════════
# 4. Markdown 보고서 생성
# ═══════════════════════════════════════════════════════════════
today = datetime.date.today().isoformat()

# confusion matrix 문자열
cm_lines = []
header = "Pred→ | " + " | ".join(f"class{c}" for c in classes)
cm_lines.append(header)
cm_lines.append("-" * len(header))
for r in classes:
    row_str = f"Actual {r} | " + " | ".join(str(cm[r][c]) for c in classes)
    cm_lines.append(row_str)
cm_str = "\n".join(cm_lines)

# 결론 문장
budget_ok_onnx  = onnx_mean  < 200
budget_ok_total = total_mean < 200
budget_str = (
    f"ONNX 추론({onnx_mean:.1f}ms)과 LogMel+ONNX({total_mean:.1f}ms) 모두 "
    f"슬라이딩 윈도우 200ms 예산 내에 {'수행됨' if budget_ok_total else '수행되지 않음'}."
)

far_best = min(far_rows, key=lambda x: x["far_per_hour"])

md_content = f"""# BCResNet-t2 ONNX 추론 성능 보고서 (RK3588 ARM CPU)

## 환경
- 하드웨어: RK3588 ARM Cortex-A76
- 런타임: onnxruntime {ORT_VERSION}
- 모델: BCResNet-t2-Focal-ep110.onnx ({model_size_kb}KB)
- 날짜: {today}

## 속도 (단위: ms/call)

| 측정 항목 | 평균 | min | max | std | 반복 N |
|---|---|---|---|---|---|
| ONNX 추론 only | {onnx_mean:.2f} | {onnx_min:.2f} | {onnx_max:.2f} | {onnx_std:.2f} | {N_ONNX} |
| LogMel(numpy) + ONNX | {total_mean:.2f} | {total_min:.2f} | {total_max:.2f} | {total_std:.2f} | {N_TOTAL} |
| LogMel(numpy) only (추정) | {logmel_mean:.2f} | - | - | - | - |

> 슬라이딩 윈도우 shift = 200ms → ONNX only 예산 충족: **{'OK' if budget_ok_onnx else 'NG'}** / LogMel+ONNX 예산 충족: **{'OK' if budget_ok_total else 'NG'}**

## 정확도 (test.csv, N={n_total_samples - errors})

| 지표 | 값 |
|---|---|
| Micro Accuracy | {micro_acc*100:.2f}% |
| Macro Accuracy | {macro_acc*100:.2f}% |
| Class 0 Accuracy (non-wake) | {per_class_acc[0]*100:.2f}% ({cm[0,0]}/{cm[0].sum()}) |
| Class 1 Accuracy (wake) | {per_class_acc[1]*100:.2f}% ({cm[1,1]}/{cm[1].sum()}) |

**Confusion Matrix:**
```
{cm_str}
```
{'오류 파일: ' + str(errors) + '건 (건너뜀)' if errors else ''}

## FAR 평가 (vad_cropped/, {len(WAV_FILES_SELECTED)}파일 / 전체 {len(WAV_FILES)}파일)

| 설정 | False Alarms | Duration (min) | FAR/hour |
|---|---|---|---|
"""
for r in far_rows:
    md_content += f"| {r['name']} | {r['false_alarms']} | {r['duration_min']:.1f} | {r['far_per_hour']:.2f} |\n"

md_content += f"""
> 파라미터: EMA alpha={EMA_ALPHA}, N-of-M ({N_OF_M_N}/{N_OF_M_M}), Refractory={REFRAC_SEC}s, Threshold={THRESHOLD}

## 결론

- {budget_str}
- 200ms shift 기준 실시간성 확보: ONNX 추론({onnx_mean:.1f}ms)은 shift 예산의 **{onnx_mean/200*100:.0f}%** 사용, 여유 {200-onnx_mean:.1f}ms.
- LogMel 포함 전처리({total_mean:.1f}ms)는 shift 예산의 **{total_mean/200*100:.0f}%** 사용, 여유 {200-total_mean:.1f}ms.
- 최저 FAR 설정: **{far_best['name']}** → {far_best['far_per_hour']:.2f} FA/hour.
- LogMel(numpy) 구현은 torchaudio 대비 종속성 없이 동작하며 RK3588 ARM에서 정상 수행됨.
"""

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"\n보고서 저장 완료: {OUT_MD}")
print(md_content)
