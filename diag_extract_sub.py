"""
no-SSN 모델에서 첫 BCBlock까지만 추출 - onnx.utils.extract_model 사용
"""
import numpy as np, sys, wave, onnx
sys.path.insert(0, '.')
from inference_rknn import LogMel
from onnx import shape_inference
import onnxruntime as ort

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

model = onnx.load('BCResNet-t2-no-ssn.onnx')
model = shape_inference.infer_shapes(model)

# 추출할 중간 output names
targets = {
    # tensor_name: description
    '/backbone/cnn_head/cnn_head.0/Conv_output_0': 'after_first_conv',
    '/backbone/cnn_head/cnn_head.2/Relu_output_0': 'after_first_relu',
    '/backbone/BCBlocks.0.0/f2/f2.0/block/block.0/Conv_output_0': 'after_f2_conv',
    '/backbone/BCBlocks.0.0/Relu_output_0': 'after_block0',  # first BCBlock complete output
    '/backbone/BCBlocks.0.1/Relu_output_0': 'after_block1',  # second BCBlock
}

for tensor_name, desc in targets.items():
    try:
        sub = onnx.utils.extract_model(
            'BCResNet-t2-no-ssn.onnx',
            f'sub_{desc}.onnx',
            input_names=['input'],
            output_names=[tensor_name],
        )
        # test with onnxruntime
        sess = ort.InferenceSession(f'sub_{desc}.onnx')
        out = sess.run(None, {'input': feat})[0]
        out_z = sess.run(None, {'input': np.zeros_like(feat)})[0]
        print(f'{desc}: shape={out.shape}, '
              f'range=[{out.min():.3f},{out.max():.3f}], '
              f'const={np.allclose(out, out_z)}')
    except Exception as e:
        print(f'{desc}: ERROR {e}')
