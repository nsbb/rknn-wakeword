"""최소 모델 NPU 테스트"""
import numpy as np, os
import onnxruntime as ort
from rknnlite.api import RKNNLite

feat_c1  = np.random.randn(1,1,40,151).astype(np.float32)
feat_c16 = np.random.randn(1,16,40,151).astype(np.float32)
zeros_c1  = np.zeros_like(feat_c1)

def test(mname, feat):
    onnx_path = f'{mname}.onnx'
    rknn_path = f'{mname}.rknn'
    if not os.path.exists(rknn_path):
        print(f'{mname}: no rknn file'); return

    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {'input': feat})[0]

    rknn_lite = RKNNLite(verbose=False)
    rknn_lite.load_rknn(rknn_path)
    rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

    npu_out = rknn_lite.inference(inputs=[feat], data_format='nchw')[0]
    npu_out_z = rknn_lite.inference(inputs=[np.zeros_like(feat)], data_format='nchw')[0]
    rknn_lite.release()

    print(f'{mname}:')
    print(f'  ONNX  range=[{onnx_out.min():.3f},{onnx_out.max():.3f}]')
    print(f'  NPU   range=[{npu_out.min():.3f},{npu_out.max():.3f}]')
    print(f'  NPU   constant?: {np.allclose(npu_out, npu_out_z)}')
    print(f'  Match ONNX~NPU (atol=0.1): {np.allclose(onnx_out, npu_out, atol=0.1)}')

test('test_conv_c1', feat_c1)
test('test_conv_bn_c1', feat_c1)
test('test_conv_c16', feat_c16)
