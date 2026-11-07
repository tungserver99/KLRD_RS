# Codex Implementation Guide — KLRD-RS trên Llama-3.2-3B-Instruct

**Mục tiêu:** bổ sung Experiment 1 (E1) và Experiment 2 (E2) cho `meta-llama/Llama-3.2-3B-Instruct` bằng cách **thêm code mới có tổ chức, tái sử dụng phần toán học hiện có, và tuyệt đối không phá vỡ Qwen**. Thiết kế lớp mở rộng để lần sau thêm backbone thứ ba mà không phải copy toàn bộ pipeline.

**Trạng thái:** đặc tả giao Codex để **triển khai source code, tests, tài liệu và server launch scripts**. Không phải báo cáo kết quả thực nghiệm. Không có GPU/model run nào được phép thực hiện trên local.

**Repository:** https://github.com/tungserver99/KLRD_RS

**Ngày soạn:** 2026-11-05.

---

## 0. Đọc trước — yêu cầu quan trọng nhất

1. **Giữ nguyên mọi workflow Qwen đang hoạt động.** Không xóa/đổi tên/sửa ngầm `main.py`, `main_exp2.py`, các launcher Qwen, `gpt_guide/`, mã E3 và các experiment artifacts lịch sử. Không overwrite `experiment1_rs/`, `experiment1_rs_continuation/`, `experiment2_rs/`, `e2_inputs_v2/`, `inputs/`, các manifest/checksum đã đóng băng.
2. **Phạm vi chỉ E1 + E2 cho Llama.** Không port LoRA, E3, E3-full, ARC-C, E2B paraphrase, SaTML/Experiment 4 hoặc chạy benchmark khác. Có thể ghi chú về khả năng mở rộng sau này nhưng không implement những phần này.
3. **Giữ nguyên phương pháp KLRD-RS.** Phần toán học, test statistic, KL budget và các giả thiết thống kê không đổi. Không đổi fingerprint thành watermark kiểu logits bias/greenlist, không thay bằng DPO/CE, không tạo thêm detector/threshold theo clean data.
4. **Không chạy model tại local.** Local không có GPU. Codex chỉ đọc/sửa/viết source, static checks và unit tests thuần CPU/synthetic nhỏ. Cấm tải model weights/datasets lớn; cấm generation, inference, CUDA tests, vLLM worker tests, benchmark, PBS submission, cloud/API/paraphrase calls. Viết sẵn lệnh/scripts, **người dùng tự chạy trên server GPU**.
5. **Không tự tạo ZIP hoặc nộp kết quả thí nghiệm.** Codex bàn giao diff/source paths, tests, lệnh cho người dùng, điều kiện tiên quyết và các vấn đề còn mở. Không tuyên bố PASS khoa học nếu chưa có kết quả server.
6. **Tối thiểu hóa sửa code cũ.** Ưu tiên *additive modules + new launchers + new model profile*. Không làm cuộc đại refactor Qwen. Nếu thực sự phải đụng shared module, thay đổi phải nhỏ, backward-compatible, có regression tests và giải thích cụ thể; tuyệt đối không nới lỏng các guard để chạy qua lỗi.

**Quyết định kiến trúc:** `Qwen legacy path (giữ nguyên)` song song với `portable E1/E2 path (Llama đầu tiên)`; cả hai tái sử dụng các hàm toán độc lập với model. Không copy/paste toàn bộ `klrd_rs/e2/runner.py`/`pipeline.py` thành biến thể Llama rồi sửa chuỗi Qwen một cách thủ công.

---

## 1. Nguồn sự thật — bắt buộc đọc trước khi code

Đọc toàn bộ các file dưới đây và kiểm tra call graph thực tế trước khi thiết kế implementation:

### 1.1. Đặc tả nghiên cứu

- `gpt_guide/KLRD_RS_FINAL_METHOD_AND_EXPERIMENT_1_SPEC.md` — **nguồn sự thật về toán học E1**.
- `gpt_guide/KLRD_RS_EXPERIMENT_2_CODEX_SPEC.md` — **nguồn sự thật về exact teacher/black-box E2**.
- `gpt_guide/KLRD_RS_EXPERIMENT_2_VLLM_SPEC.md` — cơ chế exact teacher dùng vLLM.
- `README_CONTINUATION.md` và `README_EXPERIMENT2.md` — quy trình thực tế và numerical-policy quirks đã xử lý cho Qwen.
- `README_NUMERICAL_POLICY.md` và `README_VLLM_EXPERIMENT2.md` — cảnh báo số học, precision, worker runtime.

### 1.2. Code phải kiểm tra

**Toán học có thể tái sử dụng** (đừng viết lại nếu không cần):

- `klrd_rs/math.py`: `planes`, `rotate`, `keys`, `sphere_p`, `fit_fisher`, `choose_lambda`.
- `klrd_rs/readout.py`: `RotationReadout` (`log_probs`, `observed`, `full_scores`, `kl`, `local`).
- `klrd_rs/data.py`: `prompt_ids`, `encode_response`.
- `klrd_rs/diagnostics.py`: geometry, spherical null, fixed-key pilot.
- `klrd_rs/replay.py`: clean-prefix residual replay/context sampling.
- `klrd_rs/io.py`: reproducible cache, hashes, writes.

**E1/Qwen legacy**:

- `klrd_rs/config.py`, `klrd_rs/model.py`, `klrd_rs/inputs.py`, `klrd_rs/pipeline.py`, `klrd_rs/continuation.py`, `main.py`.
- `tools/import_legacy.py`: đầu vào PCA Qwen được import từ repo KLRD; **không có logic chuẩn bị clean-Llama calibration từ đầu**.

**E2/Qwen legacy**:

- `klrd_rs/e2/config.py`, `klrd_rs/e2/frozen.py`, `klrd_rs/e2/runner.py`.
- `klrd_rs/e2/sampler.py`, `klrd_rs/e2/verifier.py`, `klrd_rs/e2/cohort.py`, `klrd_rs/e2/exclusions.py`.
- `klrd_rs/e2/vllm_model.py`, `klrd_rs/e2/vllm_backend.py`, `klrd_rs/e2/vllm_runtime.py`, `klrd_rs/e2/gates.py` và các test `tests/test_exp2*.py`.
- `main_exp2.py` và script `run_experiment2.sh`.

**Chỉ đọc để xác định ranh giới (không port)**: các module `klrd_rs/e3*`, `main_exp3*.py`, E2B, Experiment 4.

### 1.3. Các coupling Qwen đã biết — không được bỏ sót

- `klrd_rs/model.py` và `klrd_rs/e2/sampler.py` chỉ cho `model_type == 'qwen2'`.
- `klrd_rs/e2/frozen.py` khóa geometry `(2048,32)`, lấy E1 từ `experiment1_rs_continuation/`, kiểm tra receipt và checksum Qwen.
- `klrd_rs/e2/vllm_model.py` kế thừa `Qwen2ForCausalLM` và có custom `RotationFinalRMSNorm`.
- `klrd_rs/e2/vllm_runtime.py` dùng `ARCHITECTURE='KLRDRSQwen2ForCausalLM'`.
- `klrd_rs/e2/vllm_backend.py` đăng ký custom Qwen architecture tại worker; `runner.py` trực tiếp tạo `E2ModelAdapter(Config())` và `E2Config()`.
- `klrd_rs/e2b/scoring.py` và E3 cũng tự dựng `Config()`/`E2Config()`. **Ngoài phạm vi triển khai**, chỉ ghi trong compatibility notes; không sửa E2B/E3 ở task này.
- Kết quả từ E1 continuation/Qwen có lịch sử numerical diagnostics riêng, không phải input cho Llama.

**Yêu cầu Codex:** ghi rõ các coupling khác tìm thấy trong code, phân loại `must change for portable E1/E2`, `Qwen-only and leave untouched`, `future E3` trước khi viết code.

---

## 2. Quy tắc toán học bất biến

Với mỗi generation prefix `c`, clean model cho **residual r(c) ngay trước final RMSNorm**. Đặt native frozen final normalization/readout là `G`, `tau=0.8`:

```text
p0(w|c) = softmax(G(r(c))/tau)[w]

Build m=32 orthonormal 2D rotation planes in a public PCA128 subspace.
A_j = q_j p_j^T - p_j q_j^T
R(a) = exp(sum_j a_j A_j)       # exact finite orthogonal rotation
phi_j(c,w) = d/da_j log p_a(w|c) at a=0
F = E_[clean contexts, w~p0] phi phi^T
L = (F_hat + eps*I)^(-1/2)     # freeze before sampling owner keys

k ~ Uniform(S^(m-1))
a_k = lambda * (L @ k)
q_k(w|c) = softmax(G(R(a_k)r(c))/tau)[w]

psi(c,w) = L @ phi(c,w)
V(D) = sum_over_visible_response_positions psi(c_t, y_t)
T_k(D) = dot(k,V)/norm(V), or p=1 if V is zero
p_k(D) = betainc((m-1)/2, (m-1)/2, (1-T_k(D))/2)
```

`L` phải lấy đúng orientation như `klrd_rs/math.py`/`readout.py` đang dùng. `lambda` là **một strength chung cho các owner keys của Llama**, được fit với calibration-only keys; không fit từng owner key và không chọn key “đẹp”.

### 2.1. Vì sao chuyển được sang Llama

Llama-3.2-3B-Instruct có pre-final residual, final RMSNorm và linear `lm_head`. Với phép xoay trực giao, `||R(a)r||=||r||`, nên denominator của RMSNorm không đổi; gain của RMSNorm vẫn được nhân **sau** khi xoay. `RotationReadout` chính xác về mặt toán học khi weight/gain/epsilon/bias lấy từ **native Llama** và đọc đúng residual.

GQA, RoPE, số layer, tied embeddings không xuất hiện như giả thiết đặc biệt trong phép xoay/readout. Tuy nhiên chúng ảnh hưởng residual distribution, nên mọi geometry dựa vào distribution phải tính lại trên Llama.

### 2.2. Phân biệt đại lượng shared và model-specific

**Giữ nguyên:** `tau=.8`, `pca_dim=128`, `m=32`, công thức orthogonal planes, spherical null, `alpha={.01,.0001}`, clean/no-threshold policy, full-vocabulary exact rotation, quy tắc text-only verification.

**Phải tạo mới cho Llama:** model/tokenizer SHA, chat template, clean responses, residual activations, PCA basis+mean, planes `P/Q`, Fisher `F`, whitener `L`, calibration strength `lambda`, Llama response-score caches và provenance/checksums. Không tái sử dụng bất kỳ geometry hay `lambda=0.09271215438842773` nào của Qwen.

**Key policy:** dùng independent calibration-key và owner-key streams; định trước seed(s), key ID ordering và owner IDs trước khi xem TPR/FPR. Khóa geometry/Fisher/lambda **trước** khi tạo final owner keys. Không lọc/resample keys dựa trên kết quả. Có thể dùng cùng `m=32` và cùng quy tắc số lượng 32 calibration + 64 validation/owner keys; dùng namespace/seed riêng cho Llama để tránh nhầm Qwen. Lưu hashes của keys; không đưa key vectors vào report hoặc Git.

### 2.3. Những giả thiết không được diễn giải quá mức

- Spherical p-value là exact theo **key-randomization null conditional on key-independent fixed `V`**; không tự động bảo đảm fixed-key FPR <=1% cho mọi clean distribution.
- E1 local advantage không chứng minh E2 free-generation TPR hay tính bền vững của learned student.
- TPR/FPR của Qwen không chuyển sang Llama theo lý thuyết. Phải đo bằng Llama output thực tế.
- Không thay đổi định lý/verifier để cứu một kết quả yếu.

---

## 3. Mục tiêu kiến trúc source code — minimal-risk, mở rộng được

### 3.1. Giữ Qwen cũ chạy đúng như trước

**Không sửa các lệnh hiện tại**:

```bash
bash run_experiment1.sh
bash run_experiment2.sh
python main.py --continue-experiment1
python main_exp2.py --help
```

Không khởi động những lệnh này trên local; đây chỉ là tính tương thích nguồn và hành vi CLI. Mọi tests cũ phải tiếp tục import và dùng đúng old defaults, old Qwen artifact paths, old SHA manifests. Không thay Qwen bằng Llama trong bất cứ default config nào.

### 3.2. Thêm lớp portable với bề mặt nhỏ

**Cấu trúc khuyến nghị (có thể điều chỉnh tối thiểu nếu call graph cho thấy tên khác hợp lý):**

```text
configs/
  model_profiles/
    llama32_3b.yaml             # new profile; dễ thêm model mới

klrd_rs/
  portable/
    __init__.py
    profile.py                  # load/validate model + experiment profile
    hf_backbone.py              # native HF final-norm adapter, generic
    calibration.py              # clean response + pre-norm activation build
    e1.py                       # new E1 orchestration reusing existing math
    frozen.py                   # model-aware E1 artifact receipt/loader
    e2.py                       # new E2 orchestration/reuse existing cohort+verifier
    vllm_adapter.py             # architecture registry factory; no Qwen mutation
    vllm_llama.py               # Llama-specific exact vLLM teacher

main_portable_e1.py             # new entrypoint
main_portable_e2.py             # new entrypoint
run_llama_e1.sh                 # server-only stage command
run_llama_e2.sh                 # server-only stage command
README_LLAMA_E1_E2.md           # operational handoff

tests/
  test_portable_profiles.py
  test_portable_hf_contract.py
  test_portable_geometry_and_keys.py
  test_portable_e2_contract.py
  test_portable_qwen_compat.py
```

Đây là **gợi ý**, không ép tạo toàn bộ file nếu có thể gom hợp lý. Không tạo một framework abstraction phức tạp với hàng chục base classes. Một `ModelProfile` có validate và một HF adapter + vLLM factory là đủ cho E1/E2. Dùng các hàm cũ `math.py`, `readout.py`, `data.py`, `diagnostics.py`, `io.py` qua import — không tạo `math_llama.py` và `readout_llama.py` giống hệt.

Nếu reuse `pipeline.run_experiment` an toàn sau khi cung cấp generic adapter/inputs, hãy reuse trực tiếp. Nếu lịch sử E1 continuation hoặc receipts hard-code Qwen khiến reuse nguy hiểm, tạo **thin portable orchestration** gọi lại các math/stage functions generic; giải thích vì sao. Không copy nguyên file `pipeline.py` hay `e2/runner.py` làm bản fork trừ khi thật sự không có cách khác, và nếu bắt buộc thì trích xuất/reuse helpers chung tối thiểu thay vì duplicating hundreds of lines.

### 3.3. Thiết kế `ModelProfile`

Profile chỉ định những tham số nền tảng và kỹ thuật. Ví dụ minh họa (Codex phải xác nhận tên fields theo implementation):

```yaml
profile_id: llama32_3b
model_id: meta-llama/Llama-3.2-3B-Instruct
revision: null                # MUST RESOLVE TO IMMUTABLE COMMIT SHA BEFORE GPU RUN
model_type: llama
hf_architecture: LlamaForCausalLM
vllm_architecture: KLRDRSLlamaForCausalLM
trust_remote_code: false

scientific:
  tau: 0.8
  pca_dim: 128
  m: 32
  fisher_ridge_fraction: 0.001
  target_mean_kl_nats: 0.005
  safeguard_p99_kl_nats: 0.05
  alphas: [0.01, 0.0001]
  calibration_keys: 32
  final_keys: 64
  owner_key_rule: predeclared_first_key
  sampling_max_new_tokens: 512
  sampling_top_p: 1.0
  sampling_unrestricted_top_k: true

data:
  dataset: HuggingFaceH4/ultrachat_200k
  dataset_revision: 8049631c405ae6576f93f445c6b8166f76f5505a
  prompt_template_policy: native_tokenizer_apply_chat_template

runtime:
  teacher_dtype: float32
  use_tf32: false
  gpu_generation_backend: vllm
  verification_backend: hf
```

**Không hard-code SHA chưa xác minh hoặc chọn revision theo phỏng đoán.** Implement bước `resolve/freeze` do người dùng chạy trên server, pin exact commit, tokenizer revision, vocab/chat template hash và environment vào run manifest trước khi tính PCA. Khi đã freeze thì resume phải đọc lại đúng cùng revision, không tự động chuyển `main`/`latest`. Llama checkpoint chính thức là gated: **người dùng** sẽ xác nhận quyền truy cập/HF login trên server; local Codex không đăng nhập, không tải weights.

Không dựa vào `hidden_size=3072` hay `vocab_size=128256` theo chuỗi model để cấp phát; lấy từ **loaded native config** và kiểm tra shape. Llama-3.2-3B-Instruct điển hình có hidden size 3072, vocabulary 128256, 28 layers, `tie_word_embeddings=true` nhưng checkpoint thực tế là thẩm quyền. `variance_epsilon`/RMSNorm gain lấy từ native module. Kiểm tra `max_position_embeddings`, không cắt prompt/truncate im lặng.

### 3.4. Backbone adapter tối thiểu

Một interface nhỏ, đủ cho E1/E2:

```python
# Interface sketch only; not production implementation.
class BackboneAdapter:
    def load_clean_reference(self, profile): ...
    def get_tokenizer(self): ...
    def get_model_type(self): ...
    def get_hidden_size(self): ...
    def get_pre_final_norm(self): ...
    def get_output_head(self): ...
    def get_norm_gain_and_eps(self): ...
    def replay_pre_norm(self, encoded_rows): ...
    def build_rotation_readout(self, P, Q, tau): ...
    def native_tail_equivalence_check(self, residuals): ...
```

Hugging Face Qwen/Llama đều thường có `model.model.norm`; đừng giả định tất cả future models như vậy. `hf_backbone.py` nên có registry của các supported family adapters với structural assertions, không accept bất kỳ model tùy ý chỉ vì có `.norm`. Validation gồm: kiến trúc supported, RMSNorm formula đúng, final native head đủ logits, gamma/epsilon, head bias optional, hidden/vocab dimensions, and `trust_remote_code=False`.

Một test bắt buộc: native `lm_head(final_norm(r))` bằng `RotationReadout.tail(r)` trên một tiny synthetic model; và `RotationReadout.log_probs(r,angles)` bằng `log_softmax(lm_head(final_norm(rotate(r,P,Q,angles)))/tau)`. Không chỉ kiểm tra shape.

---

## 4. E1 Llama — tạo geometry và khóa kết quả từ đầu

### 4.1. Không được dùng E1 continuation Qwen

Các file Qwen dưới `inputs/`, `continuation_inputs/`, `experiment1_rs_continuation/` và `e2_inputs_v2/` có source/calibration/checksum gắn Qwen. Chúng **không** được thừa kế để tạo Llama E1. Không hack `klrd_rs/e2/frozen.py` bằng cách bỏ shape check rồi đọc geometry cũ.

E1 portable cần **clean-Llama input preparation** mới:

1. Chọn các source prompts từ dataset pinned `HuggingFaceH4/ultrachat_200k`, theo deterministic seed/ID và eligibility. Kế thừa chính sách splitting/exclusion của original E1, nhưng không reuse raw Qwen completions làm clean-Llama completions.
2. Tạo **ordinary clean Llama responses**, native tokenizer/chat template, `tau=.8`, `top_p=1`, top-k unrestricted, native stop tokens, one completion/prompt. Với generation, dùng vLLM standard Llama (không rotation) để hiệu quả; encode/replay bằng HF clean reference phù hợp.
3. Lưu prompt ID, split/source, generation seed, decoded text, generated native IDs, EOS/termination/roundtrip metadata, exact model/tokenizer revision, hashes. Không sử dụng dataset reference answers thay thế clean-model responses.
4. Tính pre-final-RMSNorm activations từ clean Llama trên **các clean prefixes thuộc calibration** (never eval/held-out). Tạo PCA128 public basis và activation mean có đủ provenance. Dùng policy phù hợp original KLRD: mean-orthogonal/orthonormal basis để các sanity checks hiện có có ý nghĩa. Nếu raw centered PCA không orthogonal với mean, phải xây/validate basis trong không gian trực giao với mean, không loại bỏ assertion cho qua.
5. Dùng 32 disjoint orthonormal 2D planes `P,Q` qua `klrd_rs/math.py::planes`. `P,Q` shape `(hidden_size,32)`; với Llama thông thường `(3072,32)`; xác nhận bằng loaded config.
6. Partition clean calibration contexts tách biệt theo các vai trò Fisher/covariance/strength/local như frozen E1. Không dùng E1 diagnostics hay E2 eval để fit PCA/Fisher/lambda. Nếu lựa chọn PCA samples từ cùng clean calibration pool, mô tả chính xác sự phụ thuộc này, không gọi PCA-independent với Fisher nếu không thực tế.
7. Tính tangent scores; estimate Fisher bằng clean `p0` sampling; regularize `eps = 0.001 * trace(F_hat) / m` theo policy đã định; lưu full matrix, eigenspectrum, rank, condition numbers và `L`.
8. Chọn **một** global `lambda` trên calibration-only key bank/contexts, gần target mean KL .005 và pooled p99 <= .05. Không copy Qwen lambda, không tăng budget để cải thiện detection.
9. Sau khi geometry/Fisher/lambda đã freeze, rút 64 owner/validation keys độc lập; chọn owner index 0 trước khi biết kết quả. Nếu mọi key chưa đạt numerical/strength gates, report fail/inconclusive thay vì lọc keys.
10. Chạy E1 math/null/score/local diagnostics đã quy định và chỉ report đúng phạm vi: local teacher feasibility và statistical implementation; không tự claim E2 detection.

### 4.2. Numerical policy E1 — phải đặc biệt cẩn thận

Qwen E1 từng gặp một **finite-difference numerical false-failure ở FP32**, dẫn đến `README_CONTINUATION.md`/`klrd_rs/continuation.py`. Đừng bê nguyên logic *historical Qwen two-context anchor* sang Llama; anchor đó không tồn tại trên Llama.

Yêu cầu tối thiểu:

- FP32 native model/readout; TF32 off; full-vocabulary softmax/log-softmax đúng `tau`.
- Kiểm tra exact rotation với native tail, skew-symmetry, orthogonality, norm preservation, `sum_w p0(w|c)phi_j(c,w) = 0`.
- Đối chiếu analytic tangent với independent autograd/JVP; nếu central FD chịu FP32 rounding, dùng FP64 **diagnostic copy** và nhiều step sizes như numerical continuation để xác định nguyên nhân; **không tăng tolerance tùy tiện**.
- Tách `FP32 finite-difference diagnostics` khỏi kết luận rằng công thức sai; fail-closed nếu independent exact analytic/JVP/FP64 comparisons không đạt.
- Kiểm tra Fisher đủ rank sau đúng regularization policy, `lambda` feasible, KL finite, các key local advantages. Không bỏ gate khoa học vì model mới.
- Trước mọi GPU run, Codex phải chuẩn bị tests/smoke thật nhỏ **cho người dùng chạy trên server**. Codex không được chạy chúng ở local.

### 4.3. Dữ liệu E1 cụ thể

Ưu tiên cùng quy mô/protocol Qwen E1 để so sánh: **2,000 calibration clean responses**, cùng chính sách chia partition; descriptive clean-original null diagnostic có thể dùng **2,500 clean responses** tách biệt, được sinh mới bằng Llama. Nếu cần thay đổi source-eligibility do native Llama tokenizer, report số liệu/trường hợp bị loại; không tự thay quy mô và gọi đó là matched experiment.

**Không bắt buộc phải làm lại GPT paraphrase cho E1.** Nếu không có Llama clean-paraphrased diagnostic set, đánh dấu `NOT_AVAILABLE`, không dùng Qwen paraphrases như cùng Llama native-generation cohort; E2 original-text là mục tiêu chính.

### 4.4. Artifacts E1 tách biệt và freeze

Ví dụ:

```text
runs/llama32_3b/e1/run_001/
  provenance/
    resolved_model_profile.json
    environment.json
    source_data_manifest.json
    clean_generation_manifest.json
    model_tokenizer_identity.json
    code_sha256.json
    calibration_partitions.json
    stage_status.json
  private/
    clean_calibration.jsonl
    geometry_activation_cache.*
    context_cache/
    secrets/                  # keys, only private server files
  geometry/
    basis.npz                 # PCA basis/mean/P/Q
    fisher.npz
    whitening.npz
    checks.json
  teacher/
    global_lambda.json
    local_capacity_by_key.csv
    KL_by_key.csv
  null/
    sphere_math_tests.json
    fixed_key_pilot.json
  keys/
    key_ids_and_hashes.json   # public-ish hashes, no vectors
  report/
    experiment1_report.md
    summary.json
    e1_frozen_receipt.json
```

Nên theo cấu trúc và schema cũ khi có thể, **nhưng không để schema cũ giả định Qwen**. E1 frozen receipt phải ghi `profile_id`, immutable model/tokenizer IDs, actual hidden/vocab shapes, files SHA-256, mathematical config, keys and strength hashes. E2 chỉ được load từ receipt này; không tự fit lại geometry hay replace file.

---

## 5. E2 Llama — exact teacher vLLM + black-box verification

### 5.1. Không tái sử dụng custom Qwen model class

`klrd_rs/e2/vllm_model.py` hiện kế thừa `vllm.model_executor.models.qwen2.Qwen2ForCausalLM`. **Không đổi class này thành Llama, không sửa cách registration Qwen.**

Tạo một class Llama mới (`KLRDRSLlamaForCausalLM`) dựa trên **native `vllm.model_executor.models.llama.LlamaForCausalLM` của phiên bản vLLM đã pin**. Có thể tái sử dụng/small-extract phần `RotationFinalRMSNorm` và geometry configure routines của Qwen nếu có thể làm mà không thay đổi semantics Qwen; hoặc tạo thin Llama wrapper nếu extraction rủi ro. Tuyệt đối không instantiate Qwen class rồi cố load Llama weights.

**Bắt buộc kiểm tra source của vLLM phiên bản mục tiêu trước khi code:** constructor signature, norm module placement, forward `(hidden_states, residual)` contract, logits/head loading, weight tying, actual model registration API, logits and EOS paths, mutable buffers under CUDA Graph. Đừng mặc định vLLM `main` hiện tại giống v0.32.0.

Runtime Qwen E2 hiện pin `vllm==0.32.0`; ưu tiên thử cùng runtime phiên bản trên server để ít thay đổi nhất. Nếu native Llama custom class đòi API khác, có thể thêm **model-profile-specific runtime support matrix** và đề xuất phiên bản tương thích cho Llama, nhưng **không upgrade global requirements/runtime của Qwen**. Chốt phiên bản sau code review/SDK contract, không đoán. Không cần cài đặt tại local.

### 5.2. Chính xác vị trí rotation

HF Llama `model.model.norm` pre-forward hook nhận full residual tensor **đã cộng xong**. Với vLLM fused residual pipeline, `norm.forward(hidden, residual)` có thể nhận hai tensor và thực hiện `hidden + residual` trong native norm. Phải đảm bảo:

```text
r = hidden                     if residual is None
r = hidden + residual          otherwise   # exactly once
r_rot = rotate(r, P, Q, a_k)
normalized = native_final_rmsnorm(r_rot)
logits = native_lm_head(normalized)
```

Không xoay post-RMSNorm tensor. Không add residual hai lần. Không tự sửa embeddings/attention. Nên giữ original full-vocabulary weights/shape, including tied embeddings, và xử lý `head.bias` nếu có (Llama thông thường không có). Không inject fingerprint vào generated output sau sampling.

**Numerical equivalence** cần kiểm tra trên GPU server:

1. `angles=0`: custom native vLLM logits match clean reference within declared numerical tolerance, including standard Llama vLLM baseline. Zero angles phải có **effectively no fingerprint**.
2. `angles!=0`: custom logits/logprobs match `RotationReadout.log_probs(r,angles)` và `log_softmax(native HF head(native RMSNorm(rotate(r)))/tau)` trên actual prefill **và incremental KV decode contexts**. Ghi max/p99 deviations; chú ý hai backend có numerical differences nên định nghĩa fairness/tolerance trước.
3. `tau=.8` áp dụng **một lần**: exact teacher sampler qua `SamplingParams(temperature=.8)`, không chia temperature thêm trong model `compute_logits` nếu sampling đã thực hiện; verifier/readout dùng tau một lần ở tính toán off-line.
4. Sampling: full vocabulary, `top_p=1`, top-k unrestricted (vLLM dùng sentinel tương thích), no penalty/rerank/beam/hidden conditioning. Native stop IDs lấy từ pinned generation config, có thể là danh sách; không assume single Qwen EOS.
5. Per-prompt seeds deterministic policy, prompt/chat template, KV-cache consistency, metadata and logits integrity, no cross-request state contamination, custom owner buffers survive CUDA Graph. Nếu khó xác minh CUDA Graph, dùng eager diagnostic trên server và fail/declare scope; không âm thầm claim production graph correctness.
6. `tensor_parallel_size=1` và FP32/no TF32 như contract Qwen, trừ khi có lý do được ghi rõ. Không chuyển teacher sang BF16/quantization để tăng tốc mà thay điều kiện thực nghiệm.

### 5.3. Vocab/stop-token và canonical text-only verifier

Llama 3.2 Instruct có chat template/tokens đặc thù, e.g. end-of-message và end-of-turn. **Dùng tokenizer + generation_config của chính checkpoint đã pin** để suy ra EOS/stop semantics, không hard-code Qwen/ChatML. `pad_token_id` có thể thiếu; chọn padding cho replay cùng `attention_mask` và xác minh no leakage, không tự thêm token làm response evidence.

- Prompt = native `tokenizer.apply_chat_template([user_message], tokenize=True, add_generation_prompt=True)`; retain full native prefix IDs.
- Teacher generation có native sampled token IDs, EOS metadata và decoded text, nhưng **primary verification** phải re-encode **visible returned text** bằng clean Llama tokenizer.
- Dùng cùng `klrd_rs/data.py::encode_response` position rule: first response token predicted at `prefix_len-1`; bỏ special/hidden control tokens khỏi selected observation positions; **EOS vẫn thuộc full vocabulary xác suất teacher**, chỉ không làm quan sát verifier.
- Score bằng clean Llama reference, **không** bằng rotated teacher itself; aggregator `V(D) = sum L phi` và spherical `p` giữ nguyên.
- Native-token scoring chỉ secondary diagnostic, so sánh roundtrip, tuyệt đối không thay primary bằng oracle generated IDs.

### 5.4. E2 evaluation design — tương đương Qwen, không tinh chỉnh theo outcome

- Fresh E2 cohort target: **5,000 distinct eligible prompts**, lấy từ pinned UltraChat source với manifest/exclusion policy; tách khỏi E1 geometry/calibration, E1 diagnostic và mọi training/test exposures. Ưu tiên `test_gen`, fallback `train_gen` khi đúng source policy, báo provenance thật; đừng làm sạch bằng cách xóa những prompt khó.
- **Execution scope hiện được chốt cho Llama là pilot group0:** tạo/freeze cohort 5,000 prompt và năm group Q=1000 theo seed đã định trước, nhưng chỉ sinh group index0 (1,000 prompt) cho mỗi condition. Chạy đủ 8 owner keys (`indices 0..7`) và clean: 8,000 owner responses + 1,000 clean = **9,000 primary responses**. Không chạy stress keys.
- Group0 có đúng **một nhóm Q=1000 và mười nhóm Q=100**, các nhóm Q=100 nằm trong Q=1000 và không được tính là 10 trial độc lập ngoài nhóm cha. Full design vẫn là 5,000 prompt/condition; pilot report phải ghi `PILOT_ONLY`, `complete=false`, không trình bày như full E2.
- Chạy ordinary clean và exact teacher trên **cùng prompts**, `max_new_tokens=512`, matched sampling settings, fixed seeds, ordinary EOS, no truncation/response regeneration.
- Group at `Q=100` và `Q=1000`; dùng **disjoint prompt groups** trong từng condition. `Q=20/50` có thể là secondary/nested nhưng không nhầm các nhóm nested là independent.
- Report TPR owner, clean FPR per key, wrong-key false detection, vector norm, cosine, p-values, valid token counts, response length, EOS/truncation/roundtrip, optional utility and visited-prefix KL. Keep alpha nominal 0.01 and conservative 1e-4 exactly.
- Đánh giá fixed-key FPR theo đúng independent groups; không tuyên bố 1% FPR được chứng minh chặt từ 5 nhóm Q1000/key hoặc bằng pooling key-dependent scores.
- Không calibrate threshold từ clean outputs. Không cherry-pick owner/key, không đổi `lambda`, `m`, query budget, cohort, alpha sau khi xem TPR.

**Smoke và pilot:** launcher `bash run_llama_e1_e2.sh` chạy E1, E2 smoke diagnostic, rồi pilot group0 và report; không tự mở rộng sang bốn group còn lại. Smoke được đánh dấu diagnostic-only, không tính vào 9,000 primary responses. Pilot dùng group0 đã freeze trong cohort 5,000; không chọn lại prompt/key theo kết quả. Mọi lần mở rộng từ pilot sang full E2 cần quyết định riêng và output identity riêng.

Pilot report được lưu riêng tại `runs/llama32_3b/e2/<run_id>/pilot/report/summary.json`, mang `status=PILOT_ONLY` và `complete=false`; điều này có nghĩa pilot đã chạy xong, còn full study chưa hoàn thành. Resume dùng các generation/scoring batch đã commit trong cùng pilot output. Khi chuyển scope từ full đang chạy sang pilot, dừng tiến trình full trước; scope/profile identity mới sẽ tạo E2 run directory mới, không trộn hay xóa artifacts full cũ. Trong lúc pilot chạy, launcher in tổng batch/response đã xong và còn lại, cùng ETA cập nhật mỗi 30 giây.

---

## 6. Artifact isolation, versioning, resumability

### 6.1. New paths

```text
runs/
  llama32_3b/
    e1/run_001/       # Llama only
    e2/run_001/       # Llama only; frozen E1 receipt reference
```

Không ghi một byte nào vào Qwen result directories. Secret keys, raw model outputs và training/eval prompts nên được giữ private trên GPU server và bị Git ignore. Không commit model weights, HF token, `.env`, API credentials, private key vectors, private response text vào Git.

### 6.2. Identity + resume

- `run_id` bind `profile_id`, full pinned model/tokenizer revision, chat template/vocab hash, source dataset revision, geometry/artifact receipts, config and source hashes, execution numerical precision.
- Existing completed generation rows phải có exact prompt ID/seed/model/mode/condition/checksum. Resume chỉ làm phần còn thiếu; fail closed nếu identity drift. `fsync`/durable journals for generated batches nếu dùng pattern hiện có.
- E2 `--mode verify` hoặc `--mode report` chỉ đọc committed data; **không phát generation/training mới**.
- E2 không được tự động chạy E1 nếu thiếu geometry; báo `MISSING_FROZEN_E1` với đường dẫn và artifact hashes cần có.
- E1 không tự động overwrite frozen receipts. Muốn thay profile/science setting/model SHA/strength phải tạo `run_002`, không mutate `run_001`.
- Qwen `anchor.py` hash hay `E2Config` singleton không phải cơ chế phù hợp cho portable; thêm loader nhận explicit E1 path/profile thay vì chỉnh Qwen constants.

---

## 7. CLI/launchers cần bàn giao

**Đề xuất interface** — Codex implement thống nhất và cập nhật README bằng đúng commands thực tế:

```bash
# READ-ONLY/local-safe: xem cấu hình/kiểm tra static wiring; không GPU/download
python main_portable_e1.py --profile llama32_3b --mode inspect
python main_portable_e2.py --profile llama32_3b --mode inspect

# SERVER ONLY — có GPU, HF access, đã cài dependencies trong environment server
python main_portable_e1.py --profile llama32_3b --mode prepare \
  --output runs/llama32_3b/e1/run_001
python main_portable_e1.py --profile llama32_3b --mode run \
  --output runs/llama32_3b/e1/run_001
python main_portable_e1.py --profile llama32_3b --mode report \
  --output runs/llama32_3b/e1/run_001

# E2 chỉ bắt đầu khi E1 đã freeze hợp lệ
python main_portable_e2.py --profile llama32_3b --mode smoke \
  --e1 runs/llama32_3b/e1/run_001 --output runs/llama32_3b/e2/run_001
python main_portable_e2.py --profile llama32_3b --mode run \
  --e1 runs/llama32_3b/e1/run_001 --output runs/llama32_3b/e2/run_001
python main_portable_e2.py --profile llama32_3b --mode report \
  --e1 runs/llama32_3b/e1/run_001 --output runs/llama32_3b/e2/run_001
```

Quy định:

- `--mode inspect` phải **strictly offline**, không gọi `from_pretrained`, không tải dataset/model/tokenizer, không init vLLM/CUDA. Chỉ kiểm tra source/profile schema/sự tồn tại optional.
- `--mode prepare` E1 phải freeze model revision trước khi tạo artifact. Nếu model permission chưa có, fail rõ với hướng dẫn cho người dùng, không download thay thế model community không đúng identity.
- `--mode smoke` E2 là explicit user-run, không tự nối sang full run.
- Shell scripts chỉ gọi một phase rõ ràng; không nối E1+E2+full studies thành run dài bắt buộc. Nếu cần PBS, chuẩn bị ví dụ `run_llama_e1.pbs`, `run_llama_e2.pbs` cho người dùng sửa resources/queue/account và tự submit.
- `--help` dùng được local không import CUDA/vLLM hoặc bắt buộc các artifacts.
- Logs phải chỉ ra version, profile, run path, stage, error reason, exact command to resume.

---

## 8. Testing contract — Codex chỉ chạy static/CPU-safe tests

### 8.1. Tests có thể viết/chạy tại local

1. `ModelProfile` parse/validation, unknown architecture fail closed, invalid revision handling, path isolation, mandatory artifact identity.
2. Synthetic tiny RMSNorm+linear head, arbitrary hidden sizes (đặc biệt 2048-like/3072-like but **tiny dimensions** cho CPU), with and without tied weights, optional head bias.
3. `RotationReadout` exact logits equality with native `RMSNorm(rotate(r)) -> lm_head`; norm preservation, tangent mean-zero, analytic vs JVP/FD in FP64 tiny toy setting.
4. Fisher rank/whitening, lambda calibration contract, independent calibration and final owner streams, spherical Beta p-values, zero-vector p=1.
5. Black-box encoding: fake Llama-style chat template, first response token index, multi-EOS/special exclusion, no privileged IDs as primary, pad/attention mask metadata.
6. E1/E2 receipt/schema/checksum/corruption/resume tests, ensure geometry cross-profile mismatch triggers hard error, Qwen dirs never targeted for write.
7. vLLM registration factory mocked (`ModelRegistry` and model implementations not loaded at import). Source-level tests can check correct per-profile architecture selection; *real kernel correctness waits for server*.
8. Regression test: import old Qwen modules/Config/E2Config, confirm unchanged defaults/path/CLI schema and no changes in historical hashes. Do not run Qwen inference/download on local.
9. Test entrypoint `--help`, `--mode inspect` offline; verify they don't import CUDA/vLLM/do network operations.

### 8.2. GPU server-only acceptance checks — chuẩn bị scripts, KHÔNG tự chạy

**E1**

- Check official Llama model/tokenizer SHA/template/hashes and native final RMSNorm/head.
- Clean activation capture before norm correct at several positions; tail parity.
- 32 planes orthogonal, skew-symmetric, rotation norm preserving; FD/JVP/analytic phi agreement.
- Fisher full rank and numerical conditioning; fixed global lambda feasible under predeclared mean/p99 KL bounds; 64 independent keys generated after geometry freeze; no key filtering.
- Spherical-null tests, local teacher key signals, diagnostic clean score behaviour; truthful E1 GO/NO_GO/INCONCLUSIVE.

**E2**

- Native clean vs custom `angles=0` logits; exact teacher vs native rotated HF log-probs on prefill **and KV-cache decoding**. Check max/p99 errors and normalized probabilities.
- EOS, chat template, no double residual addition, correct temperature application, no vocabulary truncation; native IDs/text roundtrip.
- CUDA Graph mutable rotation key A→B→A correctness (or explicitly eager-only proof and separate graph limitation), isolation between requests.
- Small smoke generation, verifier text-only score, clean and wrong-key controls, resume interruption; then user can choose full E2.

**Codex không được fabricate PASS claims, p-values hay TPR numbers.** Report unresolved server-only checks as `NOT_RUN`, with exact command and expected evidence path.

---

## 9. Acceptance criteria (source-code delivery)

Codex chỉ coi task source hoàn thành khi đạt **tất cả**:

- [ ] Old Qwen imports/CLI/config defaults/paths preserved; không xóa/rename/modifying Qwen artifacts.
- [ ] Llama config/profile explicit, pinned-revision workflow, future third-model extension documented.
- [ ] E1 có độc lập clean-Llama data + activation/PCA/Fisher/lambda fitting; không còn dependency vào Qwen E1 frozen receipt.
- [ ] E2 load **chỉ** frozen Llama E1, không tự fit lại.
- [ ] Llama exact vLLM backend is a **new model class**, không patch Qwen's registered architecture.
- [ ] Reuse original `math.py`, `readout.py`, `data.py`, spherical scoring; no algorithm changes.
- [ ] Correct native Llama final normalization/head, EOS/chat/tokenization and canonical returned-text verification, with actual server parity checks prepared.
- [ ] Reproducible, isolated Llama run directories, checksums, resume and human-readable fail-fast errors.
- [ ] CPU/static tests pass locally **without any model/dataset download**; CUDA tests prepared only.
- [ ] `README_LLAMA_E1_E2.md` includes exact sequential server commands, required env, outputs and troubleshooting.
- [ ] No E3/LoRA/full/paraphrase/benchmark implementation or execution, no ZIP packaging, no claims of scientific success.

---

## 10. Trình tự Codex nên làm (để tránh làm ẩu và sửa quá nhiều)

**Milestone 0 — Source audit, no code change:**

1. Read required files, trace E1/E2 imports, list concrete Qwen-only dependencies.
2. Decide reusable functions vs new portable wrappers; state planned touched/new files.
3. Write a short internal design note in `docs/` about compatibility constraints and server-only checks.

**Milestone 1 — Portable model/profile + correctness:**

4. Add profile/schema + HF Llama adapter; validate RMSNorm/readout math in tiny CPU tests.
5. Add isolated artifact paths and receipt validation; tests for cross-model rejection.

**Milestone 2 — Llama E1:**

6. Add clean generation/calibration input builder and PCA/public geometry fit, Fisher/lambda, independent keys and existing E1 diagnostics.
7. Add E1 CLI/launcher/docs and synthetic/static tests. Do not run real Llama E1.

**Milestone 3 — Llama E2:**

8. Implement native Llama-derived custom vLLM exact teacher and profile-aware launcher, preserving original Qwen model/registration path.
9. Implement E2 orchestration/cohort/verification from frozen Llama E1 and reuse generic scoring/controls; add mocked/static tests.
10. Add server smoke/parity scripts and `README_LLAMA_E1_E2.md`.

**Milestone 4 — Final audit/handoff:**

11. Review diff for accidental Qwen changes, hidden defaults, incorrect paths and forbidden local GPU calls.
12. Run only permissible static/CPU tests; run `--help`/offline inspect. Do not download weights or start GPU workloads.
13. Return a compact completion report: new files, any modified old files with justification, tests actually run, tests NOT_RUN (GPU), runnable server command sequence and any remaining assumptions.

---

## 11. Những lỗi cần chủ động tránh

| Sai lầm | Vì sao nguy hiểm | Phải làm |
|---|---|---|
| Chỉ đổi `Qwen` thành `Llama` trong configs | E1/Qwen geometry và e2 frozen loader vẫn dùng Qwen | Build new clean Llama E1 end-to-end |
| Reuse Qwen `P/Q/L/lambda` | Không đúng activation/Fisher distribution | Refit Llama geometry and strength |
| Nới lỏng `model_type` check thành `pass` | Có thể áp dụng sai norm/readout | Explicit supported architecture adapter + parity |
| Copy custom `Qwen2ForCausalLM` để load Llama | Model forward/weights/rotary behavior không tương thích | Derive from native vLLM `LlamaForCausalLM` |
| Xoay sau RMSNorm hoặc cộng residual 2 lần | Teacher distribution đổi | Capture/rotate at pre-final norm, parity check |
| Chia logits cho `.8` ở model **và** SamplingParams | Teacher không còn đúng distribution | Temperature once only |
| Hard-code một EOS ID | Llama native EOS may be list/end-turn aware | Read pinned model generation_config; test stop semantics |
| Dùng native generated token IDs làm verifier chính | Không còn black-box text-only | Retokenize returned text |
| Fit threshold trên clean Llama | Thay null/verification contract | Preserve analytical spherical Beta p-values |
| Tune `lambda`, chọn key dựa E2 TPR | Hỏng integrity của experiment | Freeze before E2; show failures honestly |
| Sửa Qwen `main.py`/`e2/runner.py` hàng loạt | Có thể làm hỏng baseline tốt | New portable entrypoints/wrappers |
| Tự chạy model ở local | Không có GPU, tốn tài nguyên và vi phạm phạm vi | Code + tests CPU only, server commands for user |

---

## 12. Output Codex phải trả lời tôi sau khi code

Hãy báo cáo **không dài dòng nhưng cụ thể**:

1. `Implementation summary`: giải thích mapping Qwen old path vs Llama new portable path.
2. `Created files`: file list (paths) và mục đích chính.
3. `Existing files changed`: file list hoặc `none`, với lý do và compatibility assessment.
4. `Reuse manifest`: các math/readout/verifier functions reused without scientific change.
5. `Local validation`: tests **đã thực sự chạy**, lệnh và pass/fail; không nhận là server test.
6. `Server instructions`: command từng bước E1 prepare → E1 run/report → E2 smoke → E2 run/report, yêu cầu GPU/HF/vLLM/environment; không tự thực hiện.
7. `Remaining risks`: những parity/runtime assertions phải kiểm chứng trên GPU trước khi tin kết quả.
8. `Qwen preserved`: xác nhận old defaults, paths, launchers, artifacts không bị sửa/xóa.

**Không nộp ZIP, checkpoint, bảng TPR hay scientific claims. Không yêu cầu Codex chờ server trả kết quả hoặc chạy thử với weights.**

---

## 13. Các tham chiếu chính

- Project: https://github.com/tungserver99/KLRD_RS
- Official Llama model: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- HF Llama implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- vLLM Llama model (reference; **inspect pinned tag, not only main**): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py
- Source E1 design: `gpt_guide/KLRD_RS_FINAL_METHOD_AND_EXPERIMENT_1_SPEC.md`
- Source E2 design: `gpt_guide/KLRD_RS_EXPERIMENT_2_CODEX_SPEC.md`
- Numerical and generation policies: `README_NUMERICAL_POLICY.md`, `README_VLLM_EXPERIMENT2.md`

**Final instruction to Codex:** Prioritize a correct, small additive implementation. Preserve the proven Qwen research path untouched. Enable `Llama` by adding a clean model-specific backbone and artifact identity around the *same* KLRD-RS mathematics. Build for future backbones by avoiding Qwen-shaped constants, but do not overengineer or prematurely port E3.
