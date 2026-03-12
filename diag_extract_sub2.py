"""
BCBlock 내부 세부 단계 추출 - 어느 op에서 NPU가 틀리는지 파악
"""
import numpy as np, sys, wave, onnx
sys.path.insert(0, '.')
from inference_rknn import LogMel
from onnx import shape_inference
import onnxruntime as ort

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

targets = {
    '/backbone/BCBlocks.0.0/ReduceMean_output_0': 'after_rm',           # ReduceMean depthwise conv
    '/backbone/BCBlocks.0.0/f1/f1.0/block/block.0/Conv_output_0': 'after_f1_conv',  # f1 Conv(1×3)
    '/backbone/BCBlocks.0.0/f1/f1.0/block/block.2/Sigmoid_output_0': 'after_sigmoid',
    '/backbone/BCBlocks.0.0/f1/f1.0/block/block.2/Mul_output_0': 'after_mul',  # Swish output
    '/backbone/BCBlocks.0.0/f1/f1.1/Conv_output_0': 'after_f1_1conv',   # final f1 conv
    '/backbone/BCBlocks.0.0/Add_output_0': 'after_add',                  # Add before ReLU
}

for tensor_name, desc in targets.items():
    try:
        onnx.utils.extract_model(
            'BCResNet-t2-no-ssn.onnx',
            f'sub2_{desc}.onnx',
            input_names=['input'],
            output_names=[tensor_name],
        )
        sess = ort.InferenceSession(f'sub2_{desc}.onnx')
        out = sess.run(None, {'input': feat})[0]
        print(f'{desc}: shape={out.shape}, range=[{out.min():.4f},{out.max():.4f}]')
    except Exception as e:
        print(f'{desc}: ERROR {e}')
