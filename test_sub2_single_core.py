"""NPU_CORE_0 (single core)로 테스트"""
import numpy as np, sys, wave, os
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknnlite.api import RKNNLite
import onnxruntime as ort

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]
feat2 = (feat * 0.5 + 0.3).astype(np.float32)

for desc in ['after_sigmoid', 'after_mul', 'after_add', 'after_block0']:
    rknn_path = f'sub2_{desc}.rknn' if os.path.exists(f'sub2_{desc}.rknn') else f'sub_{desc.replace("after_","after_")}.rknn'
    onnx_path = f'sub2_{desc}.onnx' if os.path.exists(f'sub2_{desc}.onnx') else f'sub_{desc}.onnx'
    if not os.path.exists(rknn_path):
        rknn_path = f'sub_{desc}.rknn'
        onnx_path = f'sub_{desc}.onnx'
    if not os.path.exists(rknn_path):
        continue

    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {'input': feat})[0]

    rknn_lite = RKNNLite(verbose=False)
    rknn_lite.load_rknn(rknn_path)
    rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)  # SINGLE CORE
    npu_out = rknn_lite.inference(inputs=[feat], data_format='nchw')[0]
    npu_out2 = rknn_lite.inference(inputs=[feat2], data_format='nchw')[0]
    rknn_lite.release()

    const = np.allclose(npu_out, npu_out2)
    match = np.allclose(onnx_out, npu_out, atol=0.05)
    print(f'{desc} [core0]: ONNX=[{onnx_out.min():.4f},{onnx_out.max():.4f}] '
          f'NPU=[{npu_out.min():.4f},{npu_out.max():.4f}] '
          f'const={const} match={match}')
