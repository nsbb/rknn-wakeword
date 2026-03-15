"""다중 BCBlock NPU 테스트"""
import numpy as np, os
import onnxruntime as ort
from rknnlite.api import RKNNLite

feat = np.random.randn(1,1,20,151).astype(np.float32)
feat2 = np.random.randn(1,1,20,151).astype(np.float32)

for n_blocks in [1, 2, 4, 8]:
    mname = f'test_stack_{n_blocks}'
    if not os.path.exists(f'../models/porting/{mname}.rknn'):
        print(f'{mname}: no rknn'); continue

    sess = ort.InferenceSession(f'../models/porting/{mname}.onnx')
    onnx_out = sess.run(None, {'input': feat})[0]

    rknn_lite = RKNNLite(verbose=False)
    rknn_lite.load_rknn(f'../models/porting/{mname}.rknn')
    rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

    npu_out  = rknn_lite.inference(inputs=[feat],  data_format='nchw')[0]
    npu_out2 = rknn_lite.inference(inputs=[feat2], data_format='nchw')[0]
    npu_out_z = rknn_lite.inference(inputs=[np.zeros_like(feat)], data_format='nchw')[0]
    rknn_lite.release()

    same12 = np.allclose(npu_out, npu_out2)
    samez  = np.allclose(npu_out, npu_out_z)
    match  = np.allclose(onnx_out, npu_out, atol=0.1)
    print(f'{mname}: ONNX=[{onnx_out.min():.3f},{onnx_out.max():.3f}] '
          f'NPU=[{npu_out.min():.3f},{npu_out.max():.3f}] '
          f'const={same12}/{samez} match={match}')
