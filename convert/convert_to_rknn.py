import os
import sys
import numpy as np
import json
import time
from datetime import datetime
from rknn.api import RKNN
import onnxruntime as ort
from typing import Optional, Dict, Any

class ModelConverter:
    """
    Design Spec 1: Model Converter
    (v1.0.2 - 통합 수정 버전: Import 누락, Shape Mismatch, Argument 불일치 해결)
    """
    def __init__(self, verbose: bool = True):
        self.rknn = RKNN(verbose=verbose)
        self.report = {
            "onnx_model": "",
            "rknn_model": "",
            "quantization": "",
            "input_shape": [],
            "output_shape": [],
            "conversion_time_sec": 0,
            "model_size_mb": {"onnx": 0, "rknn": 0},
            "validation": {"max_diff": 0, "mean_diff": 0, "passed": False},
            "timestamp": ""
        }

    def convert_onnx_to_rknn(
        self,
        onnx_path: str,
        rknn_path: str,
        quantization: str = "fp16",
        dataset_path: Optional[str] = None
    ) -> bool:
        start_time = time.time()
        do_quant = (quantization == "int8")
        
        print(f'\n--> [v1.0.2] Starting conversion: {onnx_path}')
        
        # 1. Config
        print('--> Config model')
        self.rknn.config(
            target_platform='rk3588',
            mean_values=[[0]],
            std_values=[[1]]
        )
        print('done')

        # 2. Load
        print('--> Loading model')
        if not os.path.exists(onnx_path):
            print(f"Error: ONNX file not found at {onnx_path}")
            return False
        ret = self.rknn.load_onnx(model=onnx_path)
        if ret != 0:
            print("Load model failed!")
            return False
        print('done')

        # 3. Build
        print('--> Building model')
        ret = self.rknn.build(do_quantization=do_quant, dataset=dataset_path)
        if ret != 0:
            print("Build model failed!")
            return False
        print('done')

        # 4. Export
        print('--> Export rknn model')
        ret = self.rknn.export_rknn(rknn_path)
        if ret != 0:
            print("Export rknn model failed!")
            return False
        print('done')

        # Update metadata
        self.report["onnx_model"] = os.path.basename(onnx_path)
        self.report["rknn_model"] = os.path.basename(rknn_path)
        self.report["quantization"] = quantization
        self.report["model_size_mb"]["onnx"] = os.path.getsize(onnx_path) / (1024 * 1024)
        if os.path.exists(rknn_path):
            self.report["model_size_mb"]["rknn"] = os.path.getsize(rknn_path) / (1024 * 1024)

        # 5. Numerical Validation (Property 14)
        self.validate_conversion(onnx_path)
        
        self.report["conversion_time_sec"] = time.time() - start_time
        self.generate_conversion_report(rknn_path)
        
        self.rknn.release()
        return True

    def validate_conversion(self, onnx_path: str):
        """Compare ONNX vs RKNN outputs using NCHW format"""
        dummy_input = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)
        
        # ONNX Inference
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        outputs_onnx = sess.run(None, {input_name: dummy_input})
        
        # RKNN Inference
        ret = self.rknn.init_runtime()
        if ret == 0:
            # IMPORTANT: data_format='nchw' is required for (1,1,40,151) shape
            outputs_rknn = self.rknn.inference(inputs=[dummy_input], data_format='nchw')
            
            mse = np.mean((outputs_onnx[0] - outputs_rknn[0])**2)
            max_diff = np.max(np.abs(outputs_onnx[0] - outputs_rknn[0]))
            
            self.report["validation"] = {
                "max_diff": float(max_diff),
                "mean_diff": float(mse),
                "passed": bool(max_diff < 1e-4)
            }
            print(f"Validation Result: Max Diff={max_diff:.6f}, MSE={mse:.6f}")
        else:
            print("Runtime init failed, skipping numerical validation.")

    def generate_conversion_report(self, rknn_path: str):
        self.report["timestamp"] = datetime.now().isoformat()
        report_path = rknn_path.replace(".rknn", "_report.json")
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=4)
        print(f"Conversion report saved to: {report_path}")

if __name__ == '__main__':
    converter = ModelConverter()
    onnx_path = '../models/BCResNet-t2-Focal-ep110.onnx'
    rknn_path = '../models/porting/BCResNet-t2-Focal-ep110.rknn'
    
    if os.path.exists(onnx_path):
        success = converter.convert_onnx_to_rknn(onnx_path, rknn_path)
        if success:
            print(f"Successfully converted {onnx_path} to {rknn_path}")
    else:
        print(f"ONNX model {onnx_path} not found.")




