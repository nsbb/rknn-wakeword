"""BCBlock 패턴 NPU 테스트"""
import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite

C, H, W = 16, 20, 151
feat = np.random.randn(1,1,H,W).astype(np.float32)

sess = ort.InferenceSession('test_bcblock.onnx')
onnx_out = sess.run(None, {'input': feat})[0]
print(f'ONNX: range=[{onnx_out.min():.3f},{onnx_out.max():.3f}]')

rknn_lite = RKNNLite(verbose=False)
rknn_lite.load_rknn('test_bcblock.rknn')
rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

npu_out = rknn_lite.inference(inputs=[feat], data_format='nchw')[0]
npu_out_z = rknn_lite.inference(inputs=[np.zeros_like(feat)], data_format='nchw')[0]
rknn_lite.release()

print(f'NPU:  range=[{npu_out.min():.3f},{npu_out.max():.3f}]')
print(f'NPU zeros: range=[{npu_out_z.min():.3f},{npu_out_z.max():.3f}]')
print(f'Constant?: {np.allclose(npu_out, npu_out_z)}')
print(f'Match (atol=0.1): {np.allclose(onnx_out, npu_out, atol=0.1)}')
