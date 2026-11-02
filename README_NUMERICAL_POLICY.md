# E2 — Numerical validation policy v3

**HISTORICAL / OPTIONAL ONLY.** Full E2 no longer invokes this calibration policy
or requires any numerical PASS report. No replacement calibration policy is used.
Default output is run_010. Run `bash run_experiment2.sh` directly on the server;
the steps and acceptance budgets below describe retained optional diagnostics,
not production prerequisites. Source changes have not been executed locally.

**Trạng thái: code/tests đã viết; chưa chạy tests, model hoặc benchmark ở local.
Chưa có PASS thực nghiệm cho policy mới.** Generation vẫn là custom vLLM 0.10.2;
không viết lại backend, không thêm FP64 Linear cho generation, không HF generation.

## Thay đổi chính

### Vì sao calibration v2 dừng và v3 đổi gì?

Run_008 dừng trên baseline native, chưa đánh giá keyed teacher. Native maximum
significant-token log-error khoảng 0.001424 vượt cap v2 0.000977, trong khi TV
khoảng 0.00004954 và KL khoảng 6.89e-9 đều đạt. Cap riêng cho relative log-error
là một engineering assumption chưa được xác nhận, không phải lỗi teacher đã
được chứng minh. V3 không tăng cap đó theo giá trị fail: giữ log-error làm
diagnostic và thay veto token bằng absolute probability error. Với hai phân phối
normalized, mọi token thỏa |p_i-q_i| <= TV(p,q); token budget dùng đúng TV budget.
Các cap TV/KL/weighted và mọi strict implementation gate giữ nguyên. Báo cáo thêm
ID, p, q và probability error của token gây significant log-error lớn nhất.
V3 loại toàn bộ calibration/holdout IDs đã khai báo bởi policy/code cũ khỏi
split mới, chỉ đọc design metadata, không dùng kết quả để chọn data. Giữ nguyên
run_008 và mọi bằng chứng fail. Chưa có kết quả server cho v3, không bảo đảm PASS.

Không còn dùng một absolute log-probability tolerance cho mọi loại so sánh.
Policy phân biệt đúng implementation trên cùng context với sai khác số học khi
thay đổi matrix shape hoặc backend. Scientific thresholds/keys/lambda/P/Q/L,
sampling, 5000 prompt, phân nhóm, verifier và tiêu chí E2 không thay đổi.

### 1. Các veto strict giữ nguyên

Max error 0.0002 và p99 0.00005 tiếp tục áp dụng cho exact rotation/readout,
raw logits, RMSNorm và actual KV decode vs fresh prefill. Kiểm tra temperature
áp dụng một lần, sampled-token log probability, vocabulary, zero-rotation KL,
normalization và seeded atomic resume vẫn bắt buộc. Native baseline không thể
bỏ qua các veto này.

Các kiểm tra dùng ít nhất 10 prefill +20 actual cached decode contexts trên
holdout mới. Kiểm tra cùng-captured-residual còn được áp dụng cho cả batch/single
trên các key 0,2,7 ở holdout. Disabled custom rotation phải khớp native gốc theo
ngưỡng strict; RMSNorm native và custom được kiểm tra riêng.

### 2. Native baseline thật, rotation disabled

Chế độ chỉ dành cho validation gọi trực tiếp original vLLM
`RMSNorm.forward(hidden, residual)`, gồm đường fused residual-add + RMSNorm.
Đây không chỉ là đặt angles=0 rồi đi qua custom rotation. Không reload model,
không đổi weights/attention/KV/LM head. Worker xác nhận native mode, cấm dùng với
key khác None và tự reset mode trong finally. Đường production mặc định không đổi.

### 3. Calibration và dữ liệu mới

Trước khi đo, ghi `tests/numerical_policy_spec.json`: version, guardrails,
selection, code hashes, package/device/precision, model revision, geometry hashes,
key-bank hash và input-token hashes.

- Bỏ mười prompt numerical cũ (các prompt đầu frozen pool).
- Bỏ cả calibration/holdout IDs đã khai báo trong các policy/code cũ.
- Chọn 32 prompt calibration +32 prompt holdout, không trùng nhau, từ frozen pool.
- Chia thành bốn quartile độ dài, chọn 8+8 theo thứ tự SHA256 đã định trước trong
  mỗi quartile; interleave quartiles để batch8 chứa nhiều độ dài.
- Mỗi prompt có ba prefix chẩn đoán native: prompt, +1, +2 native sampled tokens.
  Probe có thể bỏ EOS stopping để kiểm tra context; production EOS không đổi.
- Đây là **context numerical mới**, không phải tuyên bố model chưa thấy dữ liệu
  trong pretraining; không thay cohort/group E2. Diagnostic tokens không là trials.

Calibration chỉ đo native original vLLM không key: batch8-vs-single, và native
batch/single-vs-HF FP32 trên cùng prefixes. HF chỉ replay, không sinh response.
HF Linear tạm về phép tính FP32 gốc trong reference, rồi restore policy verifier
cũ. Không tính ownership scores/p-values để fit numerical envelope.

Envelope native được khóa trước mọi keyed holdout. Resume đọc lại checksum và
binding, không fit lại; đổi code/engine/inputs phải dùng run mới. Holdout native
và các key 0,2,7 sau đó phải đạt envelope đã khóa. Không chọn lại data/key hay nới
envelope khi holdout fail. Coverage được kiểm tra theo toàn bộ tích Cartesian
prompt × prefix × key × comparison; mẫu thiếu/trùng bị từ chối.

## Diagnostics và numerical acceptance

Các log probabilities gốc phải finite và normalized theo tolerance cũ trước.
Chỉ phép **đo** distribution dùng FP64 re-normalization để tránh cancellation;
không dùng FP64 Linear hoặc thay distribution mà vLLM thực tế sample.

Ghi cho từng comparison:

- Total variation: 0.5 Σ|p−q|.
- KL(p||q) và KL(q||p).
- Symmetric probability-weighted abs log-error: 0.5 Σ(p+q)|log p−log q|.
- Max/p99 log-error trên significant tokens: union p>=1/V hoặc q>=1/V, luôn gồm
  argmax hai phía. Báo count và probability mass được bao phủ; đây là diagnostics.
- Significant-token max absolute probability error; token có log-error lớn nhất
  được ghi ID, p, q và absolute probability error để giải thích độ lớn tương đối.
- Full-vocabulary raw max/p99 vẫn báo cáo, nhưng không là veto cho cross-shape
  và cross-backend. Cùng-context exact tests vẫn dùng ngưỡng strict cũ.

### Guardrails độc lập, không fit theo failed values cũ

Lấy FP32 unit roundoff u=2^-24. Những con số sau là **engineering budgets được
khai báo trước**, không phải theorem rằng mọi phép tính FP32 phải đạt:

| Metric | Absolute guardrail |
|---|---:|
| TV | sqrt(u) = 0.000244140625 |
| Mỗi chiều KL | 2u = 1.1920928955078125e-7 |
| Symmetric probability-weighted abs log-error | 2sqrt(u) = 0.00048828125 |
| Significant-token max probability error | sqrt(u) = 0.000244140625 |

Động cơ toán học: với sai số phân phối nhỏ, TV/weighted log-error là bậc một,
KL là bậc hai. Nếu **mọi conditional prefix** của hai autoregressive laws đều
thỏa KL<=2u cùng một chiều, chain rule và Pinsker cho length512 path-TV không quá
sqrt(512u) ≈ 0.00552. Các tests hữu hạn ở đây **không chứng minh giả thiết uniform
đó**, không đo hết 512 generation steps và không chứng nhận ownership error/FPR.
Không lấy bound này thay scientific threshold hoặc kết luận không có ảnh hưởng
đến ownership. KL numerical ở đây không phải KL(keyed teacher || clean teacher).

### Native empirical envelope

Mỗi family batch/reference có envelope riêng:

`limit = min(absolute_guardrail, headroom * max(native_calibration_metric) + floor)`

- Headroom 2 cho metric bậc một, 4 cho KL gần equality.
- Floors: TV 64u, weighted 128u, significant probability error 64u, KL 2(64u)^2.
- Đây là empirical headroom + arithmetic floor, **không phải** KL triangle theorem.
- Native calibration vượt absolute guardrail thì dừng; baseline nhiễu lớn không
  thể tự hợp thức hóa một teacher sai tùy ý.
- Mỗi native/keyed held-out pair phải đạt cả envelope và absolute guardrail.
- Envelope/data/code/environment được khóa bằng checksum; không expand theo
  holdout hoặc keyed errors. Giả thiết envelope tổng quát hóa được kiểm tra trên
  holdout độc lập, không khẳng định phổ quát cho mọi future prefix.

## Trạng thái và các file cần đọc

- `exact_sampler_parity.json`, `kv_cache_parity.json`: veto strict.
- `native_fp32_calibration.json`: envelope và toàn bộ native evidence đã khóa.
- `native_fp32_holdout.json`: native baseline trên dữ liệu độc lập.
- `native_zero_rotation_parity.json`: custom zero vs original native.
- `batch_prefill_parity.json`, `reference_engine_parity.json`: distribution
  acceptance + per-context evidence, kèm full-vocab raw errors để review.
- `resume_integrity.json`: fixed-batch seeded reproduction/atomic reopen.
- `numerical_policy_acceptance.json`: ghi cuối cùng khi tất cả đạt; chứa SHA256
  của các reports và binding/calibration. Bốn PASS cũ riêng lẻ không đủ approval.

`NATIVE_CALIBRATION_INVALID`/`NATIVE_BASELINE_UNSTABLE` nghĩa là native baseline
chưa đủ ổn định theo policy đã chốt; không tự quy thành lỗi KLRD-RS. Strict failure
hoặc keyed holdout mismatch vẫn dừng, không refit. Numerical PASS chỉ cho phép
chạy tiếp E2, không phải scientific GO của phương pháp.

## Các lệnh bạn tự chạy trên Colab/server

Đồng bộ toàn bộ `klrd_rs/e2/`, `main_exp2.py`, `tools/check_vllm_exp2.py`, tests và
README. Giữ nguyên môi trường vLLM 0.10.2 đã dùng, không cần đổi backend/cài lại
chỉ vì policy. Không có lệnh nào dưới đây được Codex thực thi ở local.

```python
!python -m pytest -q tests/test_e2_numerical_policy.py tests/test_exp2_vllm_contract.py
!python tools/check_vllm_exp2.py --mode parity
```

Report của script nằm trong `verification/vllm_server/<unique-id>/`. Nếu PASS,
có thể đo throughput bằng `--mode benchmark --jobs 2`. Thay numerical budgets,
headroom hoặc sample split sau khi nhìn holdout failure là policy version mới,
cần review và dữ liệu holdout mới; không sửa để tiếp tục một run cũ.

Khi sẵn sàng chạy study:

```python
!bash run_experiment2.sh
```

File sh vẫn chỉ có `python main_exp2.py`. Run mới mặc định `experiment2_rs/run_009`.
Runner tự khóa native calibration, kiểm tra strict + fresh holdout rồi mới chạy
Stage0/pilot/full E2 nếu được numerical approval. Kết quả cũ 001–008 giữ nguyên.
Không tự chuyển Experiment3, training hoặc paraphrase API.
