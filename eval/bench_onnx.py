"""
보드 CPU에서 ONNX 추론 속도 측정
"""
import numpy as np, time, sys, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_rknn import LogMel
import onnxruntime as ort

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
session = ort.InferenceSession('BCResNet-t2-Focal-ep110.onnx', sess_options)
iname = session.get_inputs()[0].name

# warmup
for _ in range(5):
    session.run(None, {iname: feat})

# LogMel + ONNX 통합 속도 (실제 사용 시나리오)
N = 100
t0 = time.perf_counter()
for _ in range(N):
    out = session.run(None, {iname: feat})[0]
t1 = time.perf_counter()

ms_per_inf = (t1 - t0) / N * 1000
print(f'ONNX inference only:  {ms_per_inf:.2f} ms/call')
print(f'Sliding window shift: 200ms → budget OK: {ms_per_inf < 200}')

# LogMel 포함
t0 = time.perf_counter()
for _ in range(N):
    f = LogMel()(audio)[np.newaxis, np.newaxis, ...]
    out = session.run(None, {iname: f})[0]
t1 = time.perf_counter()
ms_total = (t1 - t0) / N * 1000
print(f'LogMel + ONNX total:  {ms_total:.2f} ms/call')

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()
print('probs:', softmax(out.squeeze()))
