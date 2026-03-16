# BCResNet-t2 Wake Word Detection on RK3588 NPU

BCResNet-t2 웨이크워드 감지 모델을 RK3588 NPU에서 실행하기 위한 포팅 작업 결과물.

원본 ONNX 모델을 그대로 변환하면 NPU에서 상수 출력이 발생하는 문제를 ONNX 그래프 수준에서 우회하여 해결했다. ONNX CPU와 동일한 98.68% 정확도를 NPU에서 달성했으며, 118분 TV 뉴스 배경음 기준 FAR = 0.00/hr (threshold=0.55 + EMA).

---

## 성능 요약

| 항목 | 값 |
|------|-----|
| 정확도 (test set, N=1,897) | **98.68%** |
| Wake Recall | 93.95% (264/281) |
| Non-Wake Specificity | 99.50% (1608/1616) |
| E2E 레이턴시 (LogMel + NPU) | **21.30 ms** |
| 실시간 여유 (hop 160ms 기준) | **7.5배** |
| FAR @ threshold=0.55 + EMA | **0.00/hr** (118분 기준) |
| 추천 threshold | **0.55** |

---

## 핵심 문제 및 해결

원본 ONNX를 RKNN으로 변환하면 NPU에서 3가지 버그로 인해 상수 출력이 발생한다. 모두 ONNX 그래프 수정으로 우회했다.

| # | 버그 | 원인 | 우회법 |
|---|------|------|--------|
| 1 | `ReduceMean` 미지원 | NPU 백엔드 미구현 | `depthwise Conv`로 교체 |
| 2 | H=1 중간 Conv 실패 | NPU 스케줄러 제한 | `Pad(H=1→4)` + `Slice(row 0)` |
| 3 | `AddReLU` broadcast 버그 | RKNN 퓨전 옵티마이저 버그 | `Expand`로 사전 확장 |

자세한 내용: [`docs/01_rknn_bugs_root_cause.md`](docs/01_rknn_bugs_root_cause.md)

---

## 프로젝트 구조

```
├── test_quick.py              # 단일 WAV 웨이크워드 테스트 (가장 간단)
├── inference_rknn.py          # 전체 평가 (NPU 추론, 정확도, FAR)
├── inference.py               # ONNX CPU 추론
├── fix_rknn_graph.py          # ONNX 그래프 수정 (3가지 버그 우회)
│
├── models/                    # 모델 파일
│   ├── BCResNet-t2-Focal-ep110.onnx   # 원본 ONNX
│   ├── BCResNet-t2-npu-fixed.onnx     # NPU용 수정 ONNX
│   ├── BCResNet-t2-npu-fixed.rknn     # 최종 RKNN (프로덕션)
│   └── porting/               # 포팅 과정 중간 산출물 (서브그래프, 변형 등)
│
├── eval/                      # 벤치마크 & 평가
│   ├── bench_e2e.py           # E2E 레이턴시 (LogMel + NPU)
│   ├── bench_npu.py           # NPU 추론 레이턴시
│   ├── bench_onnx.py          # ONNX CPU 레이턴시
│   ├── threshold_sweep.py     # threshold 최적화
│   └── measure_far_npu.py     # 장시간 배경음 FAR 측정
│
├── convert/                   # 변환 스크립트
├── test/                      # 서브그래프 NPU 테스트
├── diag/                      # 포팅 과정 진단/디버깅 스크립트
└── docs/                      # 문서
```

---

## 빠른 테스트

```bash
# 단일 WAV 파일로 웨이크워드 감지 테스트
conda run -n RKNN-Toolkit2 python test_quick.py <wav_file>

# 기본 테스트 파일 사용 (웨이크워드 포함)
conda run -n RKNN-Toolkit2 python test_quick.py
# → 웨이크워드 확률: 0.9223 (threshold: 0.55)
# → 결과: 감지됨
# → NPU 추론: 8.7ms
```

**테스트 파일 규칙:**
- 16kHz WAV (다른 샘플레이트도 자동 변환)
- 1.5초 이하 (초과 시 앞부분 잘림)
- `wallpad_HiWonder_251113/{speaker}/{speaker}_{label}_{idx}.wav` (label: 0=non-wake, 1=wake)

---

## 빌드 및 전체 평가

```bash
# 1. ONNX 그래프 수정
conda run -n RKNN-Toolkit2 python fix_rknn_graph.py

# 2. RKNN 변환
conda run -n RKNN-Toolkit2 python convert/convert_fixed_only.py

# 3. NPU 동작 확인
conda run -n RKNN-Toolkit2 python test/test_npu_fixed.py

# 4. 전체 평가 (정확도 + FAR)
conda run -n RKNN-Toolkit2 python inference_rknn.py

# 5. 벤치마크
conda run -n RKNN-Toolkit2 python eval/bench_e2e.py

# 6. FAR 측정 (measure_FA/ 디렉토리 필요)
conda run -n RKNN-Toolkit2 python eval/measure_far_npu.py
```

---

## 문서

| 문서 | 내용 |
|------|------|
| [`docs/01_rknn_bugs_root_cause.md`](docs/01_rknn_bugs_root_cause.md) | NPU 포팅 실패 원인 분석 (3가지 버그) |
| [`docs/02_fix_solution.md`](docs/02_fix_solution.md) | ONNX 그래프 수정 방법 및 구현 |
| [`docs/03_test_results.md`](docs/03_test_results.md) | 수정 전후 테스트 결과 비교 |
| [`docs/04_environment_setup.md`](docs/04_environment_setup.md) | 환경 설정 및 재현 방법 |
| [`docs/benchmark_results.md`](docs/benchmark_results.md) | 전체 성능 수치 요약 |

---

## 환경

- RK3588, RKNN-Toolkit2 v2.3.2, Python 3.8
- conda env: `RKNN-Toolkit2`
