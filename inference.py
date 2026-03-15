"""
ONNX 모델 export 및 4가지 FAR 설정 비교 평가
- Raw (no processing)
- Refractory only
- Refractory + EMA
- Refractory + EMA + N-of-M
"""

import os
import glob
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional
from collections import deque
import time
import matplotlib.pyplot as plt
import gc

import torch
import torchaudio

import onnxruntime as ort

from sklearn.metrics import confusion_matrix, accuracy_score

class LogMel:
    def __init__(
        self, device, sample_rate=16000, hop_length=160, win_length=480, n_fft=512, n_mels=40, apply_preemph=False
    ):  
        self.preemph = apply_preemph
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.device = device

    def __call__(self, x):
        if self.preemph:
            pre = torchaudio.transforms.Preemphasis(coeff=0.97).to(self.device)
            x = pre(x)
        self.mel = self.mel.to(self.device)
        output = (self.mel(x) + 1e-6).log()
        return output

def sliding_windows(
    audio: torch.Tensor,
    sr: int,
    win_sec: float = 1.0,
    shift_sec: float = 0.1,
    pad_end: bool = True,
    pad_value: float = 0.0
) -> Iterator[Tuple[float, float, torch.Tensor]]:
    """
    waveform에서 win_sec 윈도우를 shift_sec씩 이동.
    """
    assert audio.dim() == 1, "audio must be 1D mono tensor."

    win_samples = int(round(win_sec * sr))
    hop_samples = int(round(shift_sec * sr))
    win_samples = max(win_samples, 1)
    hop_samples = max(hop_samples, 1)

    N = audio.numel()
    if N == 0:
        return

    i = 0
    while True:
        start = i * hop_samples
        if start > N - win_samples:
            if pad_end and start < N:
                tail = audio[start:]
                pad_len = win_samples - tail.numel()
                if pad_len > 0:
                    pad = torch.full((pad_len,), pad_value, dtype=audio.dtype, device=audio.device)
                    chunk = torch.cat([tail, pad], dim=0)
                else:
                    chunk = tail[:win_samples]
                start_t = start / sr
                end_t = start_t + win_sec
                yield (start_t, end_t, chunk)
            break

        chunk = audio[start:start + win_samples]
        start_t = start / sr
        end_t = start_t + win_sec
        yield (start_t, end_t, chunk)
        i += 1


### settings ###

tau = 2 # e.g. 't8' -> 8
n_cls = 2
win_sec = 1.5
shift_sec = 0.2
threshold = 0.5

logmel = LogMel(device="cpu")

### onnx 모델 로드
onnx_path = "./models/BCResNet-t2-Focal-ep110.onnx"

print(f"⚠️Starting inference with .onnx model...")  
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
session = ort.InferenceSession(onnx_path, sess_options)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

### 정확도 평가
test_df = pd.read_csv("./test.csv")
y_true, y_pred = [], []
log_df = test_df                                                                                                              

for i in tqdm(range(len(test_df))):
    tf = test_df.iloc[i]['path']
    tf_label = int(test_df.iloc[i]['label'])
    
    waveform = torchaudio.load(tf)[0]
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 1.5초보다 짧으면 0패딩,
    # 1.5초보다 길면 앞에서 1.5초만 사용
    if waveform.shape[1] < int(1.5 * 16000):
        pad_len = int(1.5 * 16000) - waveform.shape[1]
        pad = torch.zeros((1, pad_len))
        waveform = torch.cat([waveform, pad], dim=1)
    else:
        waveform = waveform[:, :int(1.5 * 16000)]
    
    assert waveform.shape[1] == int(1.5 * 16000), "Waveform length mismatch."
    
    inputs = logmel(waveform).unsqueeze(0).numpy().astype(np.float32)
    ort_inputs = {input_name: inputs}
    outputs = session.run([output_name], ort_inputs)
    pred = torch.tensor(outputs[0])
    pred = torch.softmax(pred, dim=-1)
    prediction = torch.argmax(pred, dim=-1)

    y_true.append(tf_label)
    y_pred.append(prediction.item())
    log_df.at[i, 'pred'] = prediction.item()

true_count = sum(yt == yp for yt, yp in zip(y_true, y_pred))
micro_acc = true_count / len(y_true) * 100.0
cm = confusion_matrix(y_true, y_pred)
macro_acc = np.mean(cm.diagonal() / cm.sum(axis=1)) * 100.0
per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100.0

log_df = pd.concat([log_df, pd.DataFrame([{'path': 'Micro Accuracy', 'label': '', 'pred': f"{micro_acc:.2f}"}])], ignore_index=True)
log_df = pd.concat([log_df, pd.DataFrame([{'path': 'Macro Accuracy', 'label': '', 'pred': f"{macro_acc:.2f}"}])], ignore_index=True)
for idx, acc in enumerate(per_class_acc):
    log_df = pd.concat([log_df, pd.DataFrame([{'path': f'Class {idx} Accuracy', 'label': '', 'pred': f"{acc:.2f}"}])], ignore_index=True)
log_df.to_csv(f"{os.path.dirname(onnx_path)}/test_results.csv", index=False)


### 오탐율 평가 - 4가지 설정 비교

audio_path_list = [
    "./measure_FA/[다시보기] 뉴스1번지 (2023.12.15)  연합뉴스TV (YonhapnewsTV)_record.wav",
    "./measure_FA/체감온도 영하 5도 뚝‥강원·충북 한파특보 - [LIVE] MBC 뉴스투데이 2025년 11월 18일_record.wav",
]

# FAR 설정
EMA_ALPHA = 0.3
N_OF_M_N = 3
N_OF_M_M = 5
REFRAC_SEC = 2.0

# 4가지 설정 옵션
CONFIG_OPTIONS = [
    {"REFRACTORY_SEC": 0.0, "USE_EMA": False, "USE_N_OF_M": False, "name": "Raw (no processing)"},
    {"REFRACTORY_SEC": REFRAC_SEC, "USE_EMA": False, "USE_N_OF_M": False, "name": "Refractory only"},
    {"REFRACTORY_SEC": REFRAC_SEC, "USE_EMA": True, "USE_N_OF_M": False, "name": "Refractory + EMA"},
    {"REFRACTORY_SEC": REFRAC_SEC, "USE_EMA": True, "USE_N_OF_M": True, "name": "Refractory + EMA + N-of-M"},
]


def run_far_evaluation(audio, sr, win_sec, shift_sec, threshold, config, session, input_name, output_name, logmel):
    """주어진 설정으로 FAR 평가 실행"""
    REFRACTORY_SEC = config["REFRACTORY_SEC"]
    USE_EMA = config["USE_EMA"]
    USE_N_OF_M = config["USE_N_OF_M"]
    
    onnx_output = []
    chunk_cnt = 0
    
    # trigger state
    cooldown_until = -1e9
    false_alarms = 0
    trigger_times = []
    
    # smoothing buffers
    p_ema_state = [None]
    trigbuf = deque(maxlen=N_OF_M_M)
    
    # smoothed probs 저장용
    smoothed_probs_list = []
    
    def update_score(p):
        if USE_EMA:
            if p_ema_state[0] is None:
                p_ema_state[0] = p
            else:
                p_ema_state[0] = EMA_ALPHA * p + (1 - EMA_ALPHA) * p_ema_state[0]
            return p_ema_state[0]
        return p
    
    def check_trigger(p_smooth):
        hit = (p_smooth >= threshold)
        
        if USE_N_OF_M:
            trigbuf.append(1 if hit else 0)
            if len(trigbuf) < N_OF_M_M:
                return False
            return (sum(trigbuf) >= N_OF_M_N)
        
        return hit
    
    for (start_t, end_t, chunk) in sliding_windows(audio.squeeze(0), sr, win_sec, shift_sec):
        chunk_cnt += 1
        inputs = logmel(chunk.unsqueeze(0)).unsqueeze(0).numpy().astype(np.float32)
        ort_inputs = {input_name: inputs}
        outputs = session.run([output_name], ort_inputs)
        pred = torch.tensor(outputs[0])
        pred = torch.softmax(pred, dim=-1)
        
        onnx_output.append(pred.cpu().numpy())
        
        p = float(pred[0, 1])
        t = start_t
        
        p_s = update_score(p)
        smoothed_probs_list.append(p_s)
        fired = check_trigger(p_s)
        
        if t >= cooldown_until and fired:
            false_alarms += 1
            trigger_times.append((start_t, end_t, p_s))
            cooldown_until = t + REFRACTORY_SEC
            # 버퍼 초기화: trigger 발생 후 새로운 감지 세션 시작
            p_ema_state[0] = None
            trigbuf.clear()
    
    onnx_output_np = np.concatenate(onnx_output, axis=0)
    onnx_probs = onnx_output_np[:, 1]
    smoothed_probs = np.array(smoothed_probs_list)
    
    return onnx_probs, smoothed_probs, trigger_times, false_alarms, chunk_cnt


def plot_individual(audio, config, result, win_sec, shift_sec, threshold, save_path, config_idx):
    """각 설정 결과를 개별 plot으로 저장"""
    wav = audio
    duration = wav.shape[1] / 16000
    time_axis = np.linspace(0, duration, wav.shape[-1])
    
    onnx_probs, smoothed_probs, trigger_times, false_alarms, chunk_cnt = result
    times = np.arange(chunk_cnt) * shift_sec + (win_sec / 2.0)
    
    fig, ax1 = plt.subplots(figsize=(20, 4))
    
    # 오디오 파형
    ax1.plot(time_axis, wav.squeeze().numpy(), color="gray", alpha=0.3)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")
    
    # 확률 plot
    ax2 = ax1.twinx()
    ax2.plot(times, onnx_probs, color="red", marker="o", markersize=2, alpha=0.4, label="Raw Prob")
    
    # smoothed 확률 (EMA 사용 시)
    if config["USE_EMA"]:
        ax2.plot(times, smoothed_probs, color="orange", linewidth=2, label="Smoothed (EMA)")
    
    # trigger 시점 표시 (마커)
    if len(trigger_times) > 0:
        trigger_x = [start_t + (win_sec / 2.0) for (start_t, end_t, p_s) in trigger_times]
        trigger_y = [p_s for (start_t, end_t, p_s) in trigger_times]
        ax2.scatter(trigger_x, trigger_y, color='green', marker='v', s=100, zorder=5, 
                    edgecolors='black', linewidths=1, label='Trigger')
    
    ax2.set_ylabel("Probability", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim([-0.1, 1.1])
    ax2.axhline(y=threshold, color='blue', linestyle='--', alpha=0.7, label='Threshold')
    ax2.legend(loc='upper right', fontsize=8)
    
    # 타이틀에 설정과 trigger 횟수 표시
    title = f"{config['name']} (Triggers: {false_alarms})"
    ax1.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plot_filename = f"{save_path}_config{config_idx+1}.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    
    # 메모리 정리
    del fig, ax1, ax2
    gc.collect()
    
    print(f"    Plot saved: {plot_filename}")


# 각 오디오 파일에 대해 4가지 설정으로 평가
total_results_per_config = {i: {"false_alarms": 0, "duration": 0.0} for i in range(len(CONFIG_OPTIONS))}

for audio_path in tqdm(audio_path_list):
    title_plus = os.path.basename(audio_path)[:-4]
    print(f"\n{'='*60}")
    print(f"🔹Processing audio file: {audio_path}")

    audio, sr = torchaudio.load(audio_path)

    if audio.shape[0] > 1:
        print(f"🔹Converting to mono from {audio.shape[0]} channels.")
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sr != 16000:
        if sr > 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            print(f"🔹Resampled from {sr} to 16000 Hz.")
            sr = 16000
        else:
            raise ValueError("Sampling rate must be greater than 16kHz.")

    file_duration = audio.shape[1] / sr
    
    # 4가지 설정으로 각각 평가 및 개별 plot 저장
    save_path = os.path.join(os.path.dirname(onnx_path), title_plus)
    
    for config_idx, config in enumerate(tqdm(CONFIG_OPTIONS)):
        print(f"  ▶ Running config: {config['name']}")
        result = run_far_evaluation(audio, sr, win_sec, shift_sec, threshold, config, 
                                     session, input_name, output_name, logmel)
        
        onnx_probs, smoothed_probs, trigger_times, false_alarms, chunk_cnt = result
        print(f"    Triggers: {false_alarms}")
        
        # 개별 plot 저장
        plot_individual(audio, config, result, win_sec, shift_sec, threshold, save_path, config_idx)
        
        # 전체 통계 누적
        total_results_per_config[config_idx]["false_alarms"] += false_alarms
        total_results_per_config[config_idx]["duration"] += file_duration
        
        # 중간 메모리 정리
        del onnx_probs, smoothed_probs, result
        gc.collect()
    
    # Clean up memory
    del audio
    gc.collect()

# ========== 전체 FAR per hour 계산 (각 설정별) ==========
print(f"\n{'='*60}")
print("📊 FAR Summary (per hour) for each configuration:")
print(f"{'='*60}")
for config_idx, config in enumerate(CONFIG_OPTIONS):
    total_fa = total_results_per_config[config_idx]["false_alarms"]
    total_dur = total_results_per_config[config_idx]["duration"]
    far_per_hour = (total_fa / total_dur) * 3600.0 if total_dur > 0 else 0
    print(f"  [{config_idx+1}] {config['name']}")
    print(f"      False Alarms: {total_fa}, Duration: {total_dur/60:.1f} min, FAR: {far_per_hour:.2f}/hour")
print(f"{'='*60}")
