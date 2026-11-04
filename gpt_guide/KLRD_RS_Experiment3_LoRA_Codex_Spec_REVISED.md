# KLRD-RS — Experiment 3: LoRA Implantation by Exact Teacher KL Distillation

**Bản giao Codex — 03/11/2026, revised: mathematical and implementation contracts hardened.** Đây là đặc tả triển khai code, không phải tuyên bố thí nghiệm đã chạy. Đọc toàn bộ trước khi sửa code. Chỉ code ở máy local; mọi model run, test, generation, training, benchmark và API call do tôi thực hiện trên GPU server. **Không chạy bất kỳ unit test/CPU test nào trên local.**

## 0. Mục tiêu và phạm vi bất biến

Chúng ta đã hoàn thành: (i) Experiment 1: frozen P/Q/L, Fisher, lambda, bank 64 keys và spherical analytical verifier; (ii) Experiment 2 pilot `experiment2_rs/run_012`: exact rotated teacher sinh tự do bằng vLLM, 1.000 responses × 8 owner keys + 1.000 clean; (iii) Experiment 2B: GPT-5 mini output paraphrase, đặc biệt `experiment2_paraphrase/run_002` high-fidelity trên 400 shared prompts × 8 keys + clean. **Các kết quả trên là teacher, chưa phải weight-resident student.**

**Câu hỏi Phase 3 duy nhất:** liệu có thể distill `q_k` vào LoRA của chính Qwen2.5-3B-Instruct sao cho **student tự sinh bình thường, không có rotation/key/runtime intervention**, và fingerprint còn được xác minh qua **response text** với verifier KLRD-RS đã đóng băng?

Làm **chỉ KLRD-RS core**. Không PSS, không thay phương pháp, không thêm reward/preference/DPO/SFT vào fingerprint loss, không tìm key tốt, không thay lambda, whitening, ngưỡng p-value, prompt paraphrase, hay re-fit Fisher. Không tái sinh teacher eval 9.000/47.000 responses.

### Những nguồn phải đọc trước khi code

- `gpt_guide/KLRD_RS_FINAL_METHOD_AND_EXPERIMENT_1_SPEC.md`: định nghĩa `q_k`, KL student và spherical verifier. Phần PSS chỉ là ý tưởng cũ, **không triển khai**.
- `gpt_guide/KLRD_RS_EXPERIMENT_2_CODEX_SPEC.md`, `klrd_rs/e2/frozen.py`, `klrd_rs/e2/config.py`, `klrd_rs/e2/cohort.py`, `klrd_rs/e2/verifier.py`, `klrd_rs/readout.py`, `klrd_rs/data.py`.
- Frozen E1 trên server: `experiment1_rs_continuation/` cùng source receipt, `e2_inputs_v2/`, `inputs/calibration_manifest.json`.
- Eval E2 trên server: `experiment2_rs/run_012/` (private prompt rows, score caches, group manifest, original free-generated teacher+clean responses).
- E2B on-server: `experiment2_paraphrase/run_002/` (hoặc thư mục chạy thực tế tương ứng), đặc biệt `provenance/protocol.json`, frozen selection của 400 prompts và các paraphrases đã có. **Team-review ZIP có thể thiếu raw private responses**, ưu tiên workspace đầy đủ trên server.
- Nếu có repo KLRD cũ cạnh repo hiện tại, đọc implementation natural clean train-generation, dataset split và LoRA KD để tái sử dụng cách tổ chức dữ liệu. **Không mang verifier/rotation tham số KLRD cũ sang thay KLRD-RS**.

Tuyệt đối không tạo source/frozen artifacts giả. Khi thiếu file thực sự bắt buộc, hướng dẫn rõ file nào cần đồng bộ lên server; code có thể hoàn thiện nhưng không được kết luận đã PASS.

## 1. Frozen scientific identity

```yaml
base_model: Qwen/Qwen2.5-3B-Instruct
revision: aa8e72537993ba99e69dfaafa59ed015b17504d1
tokenizer: same pinned revision
base_teacher/student_initialization: exactly same clean checkpoint
temperature_tau: 0.8
rotation_planes_m: 32
P: read-only from E1 geometry/basis.npz
Q: read-only from E1 geometry/basis.npz
L: read-only from E1 geometry/whitening.npz
lambda: 0.09271215438842773
key_bank: E1 frozen 64 validation keys, seed 2026102803
owner_key_id: 0  # fixed deterministic first validation key; do not cherry-pick using E2/E2B TPR
analytical_alpha: [0.01, 0.0001]
```

Mặc định dùng **owner key index 0**, vì quy tắc lấy chỉ số đầu tiên đã cố định, không chọn theo kết quả paraphrase hay TPR. `load_frozen()` và key ID hashes phải được xác nhận bằng mã hiện có. Việc đã biết key0 từng có TPR cao ở E2 không phải cơ sở chọn key. Muốn replicate multi-key sau này thì tạo run độc lập, không tự chọn key tốt.

Góc xoay `a_k = lambda * L @ k`, trong đó `L` là chính **frozen regularized-Fisher inverse square-root** (`L = F_eff^{-1/2}`, `F_eff = F_hat + eps I`) đã lưu ở E1; **không** tạo một Fisher hoặc inverse square-root mới. Trong E1, `L` được xây dựng đối xứng, nên score whitening `psi = L phi` đồng nhất với khai triển log-ratio `a^T phi = lambda k^T L phi`; hãy giữ nguyên orientation và implement `klrd_rs/readout.py`/`klrd_rs/math.py` đã được chốt, không transpose tùy tiện. Teacher logits phải là chính xác **pre-final-RMSNorm residual rotation**: clean residual `r(c)` -> `R(a_k)r(c)` -> native RMSNorm -> native lm_head -> chia `tau` đúng một lần -> softmax. Dùng **`RotationReadout.log_probs(r, angles)` hiện có**. Không dùng tangent approximation làm teacher, không logits bias/greenlist/top-k distillation, không điều chỉnh nhiệt độ khác 0.8.

## 2. Cấu trúc repo và lệnh

Tạo mới, tránh thay đổi logic E1/E2/E2B:

```text
main_exp3.py
run_experiment3.sh                 # chỉ gọi python main_exp3.py --mode pilot
requirements-lora.txt
README_EXPERIMENT3.md
klrd_rs/e3/
    __init__.py
    config.py
    data.py
    teacher.py
    train.py
    evaluate.py
    generation.py
    paraphrase.py                 # chỉ wrapper vào E2B GPT-5 mini đã có, gọi theo opt-in
    report.py
    provenance.py
tests/test_exp3_contract.py       # viết source-level tests, KHÔNG chạy local
```

Có thể gộp module hợp lý, không cần tạo khung rườm rà; giữ các nhiệm vụ và CLI rõ ràng. CLI nên có:

```bash
# Pilot: prepare/reuse data, train selected key0 LoRA, validate, lưu best adapter và merged model.
python main_exp3.py --mode pilot

# Chạy full 10k/1k sau khi tôi quyết định, không tự chạy nối pilot.
python main_exp3.py --mode full --output experiment3_rs/run_002

# Chỉ sinh 1.000 student responses trên E2 held-out cohort và score.
python main_exp3.py --mode evaluate --output experiment3_rs/run_001

# Chỉ paraphrase 400 STUDENT responses bằng GPT-5 mini high-fidelity protocol, khi tôi bật.
python main_exp3.py --mode paraphrase --output experiment3_rs/run_001

# Tổng hợp lại score/report mà không gọi model/API.
python main_exp3.py --mode report --output experiment3_rs/run_001
```

Khi `--mode pilot`/`--mode full` hoàn thành, tự export best adapter và **merged_student checkpoint** từ best validation epoch; không tự generate 1.000 responses. `--mode evaluate` chỉ chạy khi merged checkpoint đã có và hash hợp lệ (nếu thiếu phải báo lỗi, không âm thầm train lại). **Không tự động gọi API**. Tách `train`, `evaluate` và `paraphrase` để chủ động tài nguyên; `run_experiment3.sh` chỉ training pilot rồi in lệnh tiếp theo. File README phải ghi chính xác hành vi, không có cài đặt package trong shell launcher.

## 3. Training dataset: clean natural contexts, không dùng E2 evaluation outputs làm training

Dùng official `HuggingFaceH4/ultrachat_200k` / `train_gen`, dataset revision đã khóa ở E1. Prompt một user turn theo pinned Qwen chat template. Chỉ giữ prompt 16–512 tokens như KLRD gốc. Mỗi prompt có **một ordinary clean-M0-generated response**; đây là nguồn *contexts* cho teacher forcing, không phải fingerprint-supervised answers.

**Ưu tiên 1:** trên server tái sử dụng đúng các corpus **clean train/validation** có sẵn từ KLRD gốc nếu có prompt IDs, raw original prompt/response, model/sampling provenance và hash đáng tin cậy. KLRD gốc dự kiến 10.000 train + 1.000 validation clean sequences. Re-tokenize theo canonical tokenizer; tuyệt đối không dùng paraphrased text làm training.

**Ưu tiên 2 khi thiếu corpus**: deterministic chọn prompts từ `train_gen` theo seed/config được lưu và sinh **một clean response/prompt** bằng clean pinned Qwen model, temperature 0.8, top_p 1, top_k unrestricted, max_new_tokens 512, ordinary EOS, seed per prompt. Có thể sử dụng vLLM **clean standard Qwen2** để tạo cache một lần; không cần vLLM custom rotation. Freeze/cache text, generated IDs và hashes để resume, không tạo lại random samples ở mỗi epoch. Không lấy đáp án của UltraChat làm tương đương clean-model-generated responses trừ khi ghi rõ đổi phân phối training (mặc định không đổi).

**Disjointness bắt buộc:** loại mọi prompt overlap theo ID, exact text hash và NFKC-normalized hash với E2 `run_012` frozen eval prompts (kể cả khi E2 đã fallback sang `train_gen`), E2B cohorts và các historical held-out/test cohorts tìm được. Prompt ở calibration E1 thuộc train_gen có thể được dùng làm training **nếu được khai báo rõ** và không thuộc held-out/evaluation; nhưng ưu tiên corpus clean train riêng đúng KLRD gốc. Tuyệt đối không sử dụng teacher/clean **outputs của E2 pilot** làm training hoặc checkpoint selection.

### Quy mô chạy

| Mode | Train clean sequences | Validation clean sequences | Mục tiêu |
|---|---:|---:|---|
| `pilot` mặc định | 2.000 | 200 | Kiểm tra khả năng truyền fingerprint, tiết kiệm thời gian |
| `full` khi được yêu cầu | 10.000 | 1.000 | Triển khai cùng phương pháp với nhiều dữ liệu hơn |

Định sẵn deterministic train/val partitions trước training. `pilot` là subset đã xác định của `full`, không được chọn theo outcome. Với mỗi sequence, concat `prompt_ids + response_ids`; token selection của KL loss là **các vị trí dự đoán response** (response first token position = prefix_len-1), không tính prompt-only positions/padding. Teacher distribution **bao gồm EOS trong vocabulary**, không truncate vocabulary. Không giả tạo EOS token khi raw text không ghi. Các response rỗng/lỗi: giữ metadata và lý do không đủ eligible KL positions, không silently regenerate hoặc chọn data theo score.

Nếu KLRD corpus cũ không đủ eligibility, báo số lượng thật, hỗ trợ deterministic completion từ train_gen; không tự co mẫu mà vẫn báo là 10k.

## 4. Training target = EXACT forward KL; không thay phương pháp

Với mỗi context `c_t` từ **clean-generated training completion**:

\[
q_k(\cdot\mid c_t)=\mathrm{softmax}(G(R(a_k)r_0(c_t))/0.8)
\]

\[
p_\theta(\cdot\mid c_t)=\mathrm{softmax}(z_\theta(c_t)/0.8)
\]

\[
\boxed{\mathcal L_{KD}=\frac{1}{N_{response\ positions}}\sum_t D_{KL}(q_k(\cdot\mid c_t)\|p_\theta(\cdot\mid c_t))}
\]

**Cực kỳ quan trọng:** loss sử dụng *full vocabulary teacher distribution*, không phải CE/SFT trên generated tokens; không sampling teacher labels, không token-logit bias approximation, không KL reverse, không DPO, không SFT/CE, không hidden-state MSE, không auxiliary paraphrase objective, không optimizer trick tự đổi target. Dùng `log_q`, `log_p`; tính `sum(exp(log_q)*(log_q-log_p))`. Normalize theo **tổng response positions** trong gradient accumulation window (đừng lấy mean mẫu khi response lengths khác nhau rồi báo là mean/token). Loss ở nhiệt độ 0.8 cho **cả teacher và student**; không nhân `T^2` tuỳ tiện vì đó đổi objective đã định nghĩa.

Teacher dùng `klrd_rs/readout.py` exact finite rotation; `readout.log_probs` đã chia temperature. Student dùng final norm/lm_head frozen; chia temperature đúng một lần. Teacher and student chạy trên **chính cùng token prefixes**, đã shift và mask đúng, không lấy generated hidden states từ E2. Teacher fully detached (no grad).

**Định nghĩa rất dễ bị code sai:** ở **Hugging Face Qwen2**, `model.model.norm.register_forward_pre_hook(...)` nhận tensor **đã cộng xong residual** từ transformer block; chỉ lấy đúng `args[0]` làm `r0` hoặc `r_theta`. **Không cộng thêm `residual` lần nữa.** Logic `hidden + residual` riêng trong custom vLLM V1 final-RMSNorm **không được copy sang HF training path**. Hook phải nhận input **trước** RMSNorm, không nhận hidden state `output_hidden_states[-1]` (đã post-norm) rồi lại áp norm lần hai.

### Đường triển khai tiết kiệm GPU (A100 40 GB)

- Ưu tiên **một resident Hugging Face backbone** được load từ clean checkpoint, gắn PEFT LoRA, tránh giữ hai bản Qwen FP32 (~2 × 12 GB) đồng thời. Với teacher forward, tắt adapter bằng API PEFT được phiên bản tương ứng hỗ trợ, chạy `torch.no_grad()`, capture **pre-final-normalization** residual `r0`; với student forward bật adapter, giữ gradient qua LoRA và lấy student residual hoặc logits.
- Cố định final RMSNorm/head và input embeddings, không cần forward qua `lm_head` full-length nếu dùng capture residual để tính full-vocab logits theo token chunks. Reuse clean frozen readout, tính `q_k` theo đúng E1.
- Chú ý PEFT wrapper attribute paths và gradient checkpointing/hook: verify source, không nhầm pre-norm/post-norm, không `.detach()` trên student representation, không sửa trực tiếp module forward nếu làm mất gradients. Enable training gradient checkpointing đúng cách (`use_cache=False`); teacher `eval()`; LoRA dropout 0 nên tương thích train/eval. Nếu activation checkpointing với frozen embeddings chặn gradient đi vào adapter, dùng cơ chế PEFT/Transformers tương thích như `enable_input_require_grads()`; **không mở khóa embeddings hoặc base weights**. Hook phải gỡ ngay sau từng forward và không được giữ graph của microbatch cũ.
- Mỗi sample batch1 có ≤512 response tokens và ≤512 prompt tokens. Compute full-vocab log_probs **theo chunks khoảng 32 selected positions** để tránh OOM, FP32 log-softmax/KL, teacher detach. **Backward một lần cho microbatch** hoặc cách tương đương không lặp transformer backward không cần thiết; không materialize tensor `[batch,seq,vocab,m]`.
- FP32 clean reference/teacher theo frozen science. Student có thể dùng **BF16 autocast cho backbone**, giữ frozen weight master FP32; **compute final RMSNorm + frozen lm_head/tail của cả q và p trong FP32 ngoài autocast**, rồi log_softmax/KL ở FP32. Chỉ `student_r.float()` khi đi vào frozen tail, **không `.detach()` student_r**. Nếu dùng thông thường `model(...).logits` trong autocast, cần kiểm tra logits chưa bị lượng tử BF16 trước khi tính KL; tốt nhất capture pre-norm student residual và tính frozen FP32 tail riêng. Đây là lựa chọn hiệu năng khi chạy trên A100; ghi rõ precision. Không silent change teacher to BF16, TF32, quantized/QLoRA hay 8-bit optimizer nếu chưa được yêu cầu.
- Chạy teacher/student tuần tự, giải phóng tensor sau bước; không đồng thời chạy vLLM generation engine và training reference trên GPU 40GB. VLLM dùng ở **giai đoạn inference**, không bắt buộc/cố nhét vào backprop.
- Nếu bộ nhớ vẫn không đủ, có quyền giảm *microbatch/chunk size* hoặc bật activation checkpointing, **không giảm vocab, đổi loss, đổi teacher hoặc cắt training data** để né OOM.

### Cấu hình LoRA và optimizer ban đầu: bám theo KLRD gốc

```yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
bias: none
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
trainable: LoRA matrices only
frozen: [all_base_weights, input_embeddings, final_norm, lm_head]
learning_rate: 1.0e-5
max_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
warmup_ratio: 0.03
weight_decay: 0.0
max_grad_norm: 1.0
student_bf16_autocast: true
teacher_fp32: true
gradient_checkpointing: true
full_vocab_logit_chunk: 32
seed: 2026110301
```

Lưu exact resolved config và package versions. **Không grid-search hyperparameters** ở giai đoạn đầu; không tăng lambda/key tuning khi loss giảm chậm.

## 5. Validation, checkpoints và resume

- Trước training báo `KL(q_k||p0)` trên validation contexts như baseline. Theo dõi train `KL(q_k||pθ)`, validation `KL(q_k||pθ)` và `KL(p0||pθ)` (utility drift) trên *teacher-forced heldout clean contexts*, số tokens, throughput và GPU memory nếu server cung cấp.
- Validate cuối mỗi epoch; checkpoint **best validation teacher KL** (không chọn theo TPR E2/E2B, paraphrase hay test). Giữ epoch 1, epoch 2 và best adapter. Có thể lưu thưa checkpoint optimizer/resume theo optimizer-step để tránh mất 1 epoch khi gián đoạn; atomic commits.
- Dùng stateful optimizer/scheduler/RNG để resume thật, tránh restart pseudo-resume; lưu code/config/input/source hashes, global optimizer step và train data permutation. Không ghi đè checkpoint hoàn chỉnh hoặc trộn hai runtime.
- Nếu validation KL không tốt hơn initialization hoặc adapter không học, báo đúng `NO_TRANSFER_EVIDENCE`/`TRAINING_INCOMPLETE` tuỳ trường hợp. Không tự đổi key, nhiệt độ, threshold hoặc train thêm vô hạn. Không tạo numerical parity gate mới; chỉ kiểm tra cơ bản logits finite, gradient LoRA nonzero, weights thay đổi, loss backprop đúng và teacher base frozen trên server.

### Control

Bắt buộc đánh giá **clean M0** (đã có E2 run_012 outputs) và **fresh/untrained LoRA student** để xem thay đổi do training. Cho phép thực hiện matched `clean teacher -> LoRA` forward-KL control với cùng train contexts/schedule để đo generic training drift; lưu ý **khi khởi tạo đúng từ p0 thì KL(p0||pθ) gần 0**, nên đây có thể là một control gần no-op, không được mô tả là một fine-tuned baseline có độ thay đổi tương ứng. Nếu cần stronger generic fine-tuning control, chỉ triển khai dưới chế độ opt-in và nhãn riêng `clean_SFT` (khác loss, không được coi là KLRD-RS ablation). Không bắt pilot chạy thêm nhiều LoRA models chỉ để tăng số baseline.

## 6. Merge adapter và inference student bình thường

Sau khi chọn best adapter bằng validation teacher KL:

1. Export PEFT adapter riêng và thực hiện merge vào **clean Qwen2.5-3B** để tạo checkpoint `merged_student/` theo cách PEFT tương thích. Chỉ LoRA trainable, final norm/head vẫn frozen.
2. Reload merged checkpoint như **ordinary `Qwen2ForCausalLM`** (standard HF/vLLM), không custom `KLRDRSQwen2ForCausalLM`, `hf_overrides`, rotation norm, key injection, special logits processor hoặc external detector ở inference.
3. Kiểm tra so sánh trên một vài fixed prompts: PEFT adapter-on vs merged checkpoint có output logits gần nhau ở cùng precision; ghi nhận một phép smoke, không biến thành gate numerical parity tốn hàng giờ. Model vẫn sinh nếu bỏ toàn bộ key files khỏi inference path.
4. Lưu `base_revision`, `adapter_checkpoint`, `merged_model_path`, `model_architecture`, precision, optimizer settings, checkpoint reason và hash. **Không copy secret key/basis/Fisher vào merged student model directory**. Owner verifier vẫn cần private key + clean model, nhưng người sử dụng student không cần.

## 7. Primary black-box evaluation — cùng E2 cohort và verifier

**Chỉ sau merge**, generate 1.000 ordinary **student** responses trên đúng 1.000 frozen group-0 E2 evaluation prompts từ `experiment2_rs/run_012` (không dùng các prompts này để training/validation). Sinh **một response/prompt** bằng standard vLLM/Transformers generation, không rotate, không key, temperature 0.8, top_p=1, unrestricted top_k, max_new_tokens=512, native EOS, per-prompt deterministic seed, không rerank/filter. Fixed prompt/tokenizer/chat template/EOS giống E2. Có thể dùng vLLM V1 standard Qwen loaded từ merged checkpoint; không ép custom FP32 teacher runner của E2. Record dtype and seed; mặc định FP32 generation để dễ đối chiếu; nếu memory buộc BF16 phải tách nhãn và không tự gọi tương đương numerically.

Scoring bằng **chính `SphericalBlackBoxVerifier.score_text_batch`** của E2: original prompt + **returned decoded response text re-tokenized** -> clean pinned reference model -> public score vector `V` -> same 64 frozen owner keys -> same Beta survival p-values; m=32, α=0.01/0.0001. Không sử dụng student logits, original sampled IDs hay hidden states cho primary black-box score; không clean-calibrate threshold, không refit Fisher, không chọn key khác sau xem results.

Group evaluation:
- Primary Q=100: 10 disjoint groups theo chính E2 frozen group0/Q100 partition; key 0 (student), teacher key0 E2 run_012 và clean E2 run_012. Q=1000: 1 descriptive aggregate group; không gọi 1 nhóm này là CI đầy đủ.
- Secondary (tính offline từ cùng 1.000 score vectors, không cần sinh thêm): Q=20 và Q=50 dùng subdivisions deterministic, report số groups và sự phụ thuộc nested. Không bootstrap/query groups chồng lặp thành độc lập.
- Owner correct-key TPR, 20 wrong keys (reuse `wrong_indices(0)`), clean false detections, raw cosine/p-values, closed-set 21-way identification nếu code hiện có hỗ trợ, số responses hợp lệ, confidence intervals theo **nhóm độc lập**.
- Teacher/student gap: compare E2 exact teacher key0 vs LoRA student trên **cùng prompt IDs**; báo rõ student response mới, không gắn cặp text như một cùng random sample.
- Nếu clean responses hiện có ở E2 verified và prompt/group IDs khớp, **reuse score caches**; không cần re-generate clean. Nếu E2 pilot chỉ có team ZIP thiếu private raw, cần đồng bộ on-server E2 workspace, không bịa data.

Minimal table:

| Model | Q | Correct-key TPR @1% | TPR @1e-4 | Clean FPR @1% | Worst wrong-key rate | Mean response length | Truncation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean M0 | 100 | n/a | n/a | ... | ... | ... | ... |
| Exact rotation teacher k0 (E2 frozen) | 100 | ... | ... | ... | ... | ... | ... |
| LoRA student k0, merged | 100 | ... | ... | ... | ... | ... | ... |

No cherry-picking: all preselected rows, failures and zero vectors included.

## 8. Paraphrase evaluation sau LoRA: high-fidelity, không tự gọi API

Paraphrase chỉ là **downstream robustness evaluation** sau khi student đã đạt original-text fingerprint signal. Cài CLI `--mode paraphrase`, **không gọi API từ `pilot`, `train` hoặc `evaluate`**.

- Lấy **đúng 400 frozen shared prompt IDs** từ `experiment2_paraphrase/run_002` high-fidelity. Từ 1.000 student outputs ở §7, trích **400 student responses mới** tương ứng; không tái sử dụng bản paraphrase của teacher làm paraphrase của student.
- Dùng **chính `gpt-5-mini-2025-08-07`** và **prompt high-fidelity đã được đóng băng** trong `experiment2_paraphrase/run_002/provenance/protocol.json`; nếu có module `klrd_rs/e2b/`, tái sử dụng nguyên transport, retry, resume, formatting. Không tự thay prompt, không dùng SCW brief prompt hoặc model khác; không để GPT-5 biết key/score.
- Chỉ khoảng **400 GPT-5 mini calls** cho student k0. Không paraphrase lại clean+teacher nếu raw E2B run_002 còn nguyên và row identity khớp: reuse frozen teacher0+clean paraphrases/vectors để so sánh. Nếu không có raw/metadata chuẩn thì báo missing, không ghép nhầm.
- Score returned-text bằng same verifier; Q=20/50/100 trên 400 rows, α=1%/1e-4; original vs paraphrased TPR, wrong/clean rates, sample count và word similarity/length ratio, semantic-quality flags. Không áp dụng phép test semantic nào để tự động chọn/loại paraphrases khi tính TPR.
- SCW brief-prompt run_001 chỉ có thể đặt vào appendix theo đúng protocol khác. Không tự gọi tiếp API cho SCW, input paraphrase hoặc additional paraphrasers.

## 9. Utility/fidelity và experiment integrity

Mặc định ghi:
- Validation `KL(q_k||pθ)` và `KL(p0||pθ)` mean, p95/p99, train-vs-val curves, train time/GPU peak, number of effective response-token positions, learning rate, optimizer steps; giữ units nats/token.
- Original free-generated text: mean/median length, visible tokens, EOS stop vs max512 truncation, empty/very short, severe repeated 4-grams, text roundtrip stats, paired selected prompt examples with clean/teacher/student, blinded human-readable comparison; không tự bỏ câu ngắn/loop.
- Nếu có tiện ích sẵn, thêm benchmark standard language understanding (MMLU, HellaSwag, ARC-Challenge, WinoGrande, TruthfulQA) với config trùng nhau **sau** khi pilot fingerprint có tín hiệu; không biến benchmark thành blocker trước free-generation validation. Đừng tự suy ra quality từ một số ví dụ.
- `ownership` claim chỉ dựa trên **student merged weight model black-box samples**; teacher E2 TPR không được nhầm thành student TPR. Mọi kết luận 1-key pilot chỉ là feasibility, không chứng nhận multi-key reliability/fixed-key FPR hay arbitrary paraphrase robustness.
- KLRD-RS method phải giữ analytical random-key null; empirical FPR trên một clean model là diagnostic, không được tuyên bố universal fixed-key guarantee.

## 10. Output, resume, packaging

Mặc định `experiment3_rs/run_001/` cho pilot, tách riêng E1/E2/E2B và full run khác. Không ghi/overwrite các thư mục lịch sử, không sửa identity/checkpoint E2 để tái sử dụng. Một cấu trúc ví dụ:

```text
experiment3_rs/run_001/
  provenance/{resolved_config.json,input_manifest.json,sha256_manifest.json,environment.json}
  private/{train_manifest.jsonl,val_manifest.jsonl,eval_responses.jsonl,score_vectors.npz,keys...}
  checkpoints/{epoch_001/,epoch_002/,best_adapter/,optimizer_state...}
  merged_student/               # weights/model config; KHÔNG có key/frozen geometry
  training/{steps.jsonl,epochs.csv,validation.csv}
  scores/{per_key_group.csv,per_response.csv,clean_wrong_controls.csv}
  paraphrase/{protocol.json,student_pairs.jsonl,metrics.csv}
  utility/{generation_summary.json,examples.md,kl_summary.csv}
  report/{summary.json,experiment3_report.md}
  KLRD_RS_experiment3_review.zip
```

- Atomic checkpoint/write, deterministic response IDs, per-step and per-output hash, resumable jobs. Không khởi chạy trùng batch hoặc gọi lại GPT-5 cho completed IDs.
- **Review ZIP** chứa code/spec/config, anonymized public tables, charts, selected safe examples và diagnostic summaries; **không chứa model weights/LoRA adapter ~GB, secret key vectors, raw private train/eval prompts/responses, OpenAI API key hoặc access tokens**. Giữ private artifacts trên server, đưa danh sách manifest paths/hashes để cần thì kiểm tra.
- `summary.json` hiển thị explicit trạng thái `TRAINING_ONLY`, `EVAL_COMPLETE`, `PARAPHRASE_COMPLETE` hoặc `INCOMPLETE`, không viết `SUCCESS` khi chưa chạy student eval. Báo cáo riêng teacher vs student vs clean; pilot Q1000 chỉ 1 group.

## 10A. IMPLEMENTATION CONTRACT — bắt buộc cho Codex trước khi viết training loop

Đây là phần giải thích **toán học -> thao tác code**, tránh một implementation chạy được nhưng học sai objective. Nếu có mâu thuẫn cách diễn đạt trong các section, ưu tiên frozen E1/E2 implementation và hợp đồng này; không thay đổi phương pháp.

### A. Đường teacher và student phải tách riêng, dù chỉ giữ một backbone trên GPU

1. `k = frozen_bank[0]`, `angles = lambda_frozen * (L @ k)`, giữ nguyên thứ tự chiều, float32 và đủ 32 tọa độ.
2. Tạo **một** clean Qwen base từ pinned revision; gắn LoRA đúng modules vào backbone; freeze tất cả non-LoRA parameters. Chỉ dùng kỹ thuật một-model/two-forwards nếu PEFT adapter-disable thực sự trả về **clean p0**, kể cả sau nhiều optimizer updates. Tắt adapter có nghĩa loại bỏ mọi tác động LoRA, không phải zero secret key vào model sinh teacher.
3. **Teacher forward:** disable LoRA -> `no_grad` -> clean base forward trên `[chat_prompt, original_clean_completion]` -> hook lấy tensor `r0` ngay trước `model.model.norm` -> gọi `RotationReadout.log_probs(r0_selected, angles)`; output `log_q` phải detached. Không đưa rotation vào backbone, không dùng E2 vLLM custom norm hoặc sampled token logprobs làm nhãn huấn luyện.
4. **Student forward:** enable LoRA -> grad-enabled forward trên **cùng** input_ids/attention_mask -> capture `r_theta` trước final norm *không detach* -> apply **unrotated** original final RMSNorm và lm_head với frozen weights -> tính `log_p = log_softmax(logits_student / 0.8)`. Không gọi `RotationReadout.log_probs(r_theta, angles)` cho student, không thêm rotation ở inference.
5. Teacher/student phải dùng **same causal prefix** tại mỗi response-prediction position. Hai forwards khác adapter state, **không khác tokens/mask**. `use_cache=False`; do not change tokenizer, chat template, padding side/position ids giữa hai forwards.
6. Tất cả full-vocab categories gồm EOS. Chỉ chọn *time steps* tương ứng **visible/recoverable response-token predictions**, đúng original KLRD và E2 black-box position policy. **EOS as a vocabulary candidate** khác với **EOS as a selected teacher-forcing time step**: không chọn thêm vị trí dự đoán chính EOS làm KL training target, ngay cả khi clean generation cache lưu native EOS. Không tự thêm EOS từ decoded text, và tuyệt đối **không loại EOS khỏi softmax vocabulary** ở những time steps hợp lệ. Giữ EOS metadata để đánh giá generation.

### B. Causal alignment cụ thể

Nếu tokenized prefix có `P` tokens và response có `R` visible tokens `y0 ... y{R-1}` thì concat `ids[0:P+R]`:

- Token `y0 = ids[P]` dùng hidden/logit tại **position `P-1`**.
- Token `yj = ids[P+j]` dùng hidden/logit tại **position `P+j-1`**.
- Selected positions = `range(P-1, P+R-1)` cho response `R` visible tokens. Padding, prompt positions khác, và mọi hidden/special token ngoài policy đều masked.
- **Distillation is distribution KL on these prefixes**. Token IDs của `yj` chỉ xác định clean prefix cho bước kế; **không** được dùng target CE/NLL loss trong E3.
- Nếu dùng batch padding, map vị trí theo từng sample và `attention_mask`; không giả định right-padding index trên mọi tensor. Khóa mã so khớp vị trí với `klrd_rs/data.py`/E2 `encode_response`.

### C. Gradient + chính xác loss aggregation

Chỉ có **một** loss target trên mỗi valid position: `sum_{w in full_vocab} exp(log_q[w]) * (log_q[w]-log_p[w])`.

Trong accumulation window với nhiều microbatches, chuẩn hóa theo tổng số valid response prediction positions **của toàn window**, không lấy mean mỗi sequence/batch rồi chia `gradient_accumulation_steps` một cách máy móc. Có thể precompute số valid positions của các microbatch trước khi chạy window:

```python
# PSEUDOCODE ONLY — Codex must adapt to current HF/PEFT APIs.
window_N = sum(valid_response_position_count(mb) for mb in window)
optimizer.zero_grad(set_to_none=True)
for mb in window:
    # Same input_ids / attention_mask for teacher and student.
    with adapters_disabled(), torch.no_grad(), fp32_teacher_forward():
        r0 = capture_HF_pre_final_RMSNorm_residual(mb)
        log_q = exact_frozen_rotation_readout(r0, frozen_angles, tau=0.8)
    with adapters_enabled(), student_backbone_autocast():
        r_theta = capture_HF_pre_final_RMSNorm_residual_with_grad(mb)
    with fp32_frozen_tail():
        log_p = unrotated_student_logprobs(r_theta.float(), tau=0.8)
    # full vocabulary; log_q detached; sum over valid response positions only
    kl_sum = (log_q.exp() * (log_q - log_p)).sum(dim=-1)[valid_mask].sum()
    (kl_sum / window_N).backward()
clip_trainable_LoRA_grad_norm_and_optimizer_step()
scheduler_step()
```

Handle last incomplete accumulation window correctly. Position chunking may stream the KL computation, but must preserve **same exact sum of full-vocab KLs** and gradients. No accidental `torch.no_grad()` on student head, no `.cpu()/.numpy()` on student loss before `.backward()`, and no reusing cached teacher `log_q` from a *different* clean prefix.

### D. Frozen tail and the numerical path

- `RotationReadout.log_probs(r0, angles)` from E1 is the **authoritative exact teacher distribution** (already includes division by τ). Don't softmax once more after τ or renormalize a truncated vocabulary.
- Student normal logits are native `lm_head(final_RMSNorm(r_theta))` from **frozen, original** tail. Keep gradient through `r_theta`, even though the norm/head weights themselves require no gradients. Dividing by `tau` for student occurs **once**.
- Use float32 for selected final-tail logits and `log_softmax`, and float32/float64 for scalar loss reporting. BF16 is allowed only in backbone forward according to documented config. Don't silently recast frozen teacher weights to BF16 or use the E2 verifier's monkeypatched FP64-Linear layers as student training backbone.
- Existing E2 verifier's `RotationReadout.observed()` and `sphere_p()` are for **verification, not the training objective**. KLRD-RS uses the same `k` to build both q and verifier, but teacher KD must not optimize verifier score, raw LLR, cosine or p-values.

### E. Minimal server sanity (one microbatch, not a new calibration campaign)

On a 1–2 example server smoke, log/inspect only the essential facts: correct selected-position indexes/counts, `q`/`p0` from same clean prefixes, initial mean teacher-vs-clean KL around the frozen strength's nats/token scale (no strict exact-value gate), adapters-disabled clean output consistent with clean base, `log_q.requires_grad=False`, `log_p.requires_grad=True`, finite loss and nonzero LoRA gradients, all base/head/norm weights unchanged after one real optimizer step. This is a **bug catch**, not an additional numerical calibration/parity experiment; no demand for full-generation logits parity or long benchmark before training.

### F. Hard provenance constraints

- Training corpus must be **clean-model-produced** ordinary responses on permitted train_gen prompts; **never** use the 1,000 frozen E2 test responses for KL training. Frozen E2 prompts remain held-out. The 2k/200 pilot is an economical subset of the 10k/1k plan, not a changed fingerprint method.
- Training adapter may contain learned information but merged student export must not contain secret key vectors, P/Q/L, custom rotation hooks or dependency on private files. Verification (owner side) legitimately loads frozen secret-key material.
- Two genuinely different failure states: `training KL improves` does **not** imply `student black-box TPR passes`. Only E3 merged-model generation + original-text verifier can establish student transfer.

---

## 11. Code-level correctness (server-only smoke, không tạo thêm numerical gates)

Code có logic và source-level test cho:

1. Token shift đúng: response token thứ nhất dự đoán từ last prompt token; prompt-only và padding masked; chưa vô tình dùng response labels làm SFT.
2. Key0 đúng frozen E1 bank hash; `a=lambda Lk`; q = **exact readout** trên clean residual trước final norm, `tau=.8` một lần, full vocab incl EOS.
3. Teacher adapter disabled/no-grad; student adapter enabled/grad; base weights + lm_head + final_norm unchanged; LoRA receives nonzero finite gradients trên ít nhất một server microbatch và `optimizer.step()` thật thay đổi weights.
4. KL(q||pθ) nonnegative up to numerical rounding, no NaN; loss normalized theo total response positions; validation uses same q and masks, no training leakage.
5. Best checkpoint chosen strictly by validation KL; reload/merge normal inference works without owner key.
6. Score bằng unchanged returned-text E2 verifier; 20 wrong keys, correct key, clean reference; same frozen groups; no threshold fit.
7. Server-only small smoke can verify 1–2 training steps, one merged output; **không yêu cầu exhaustive FP32 parity, strict batch-single identity, full-KV parity, calibration gates, microbenchmarks** trước study. Diagnostic optional không chặn nghiên cứu vì chênh float32 nhỏ.
8. Sửa missing/invalid artifact fail clearly; run identity protects outputs; API retries do not double bill completed IDs.

**Tuyệt đối không chạy bất kỳ test nào trên máy local**, kể cả pytest/CPU mocks, nếu chưa được tôi yêu cầu. Chỉ viết source và hướng dẫn server; không tải model, không cài package, không gọi API, không spin up experiment, không đóng gói results giả.

## 12. Handoff cho tôi

Sau khi hoàn thành code, Codex trả về ngắn gọn:

1. Danh sách file đã thêm/sửa, source files nào được reuse, thay đổi scientific settings (mong đợi: không có).
2. Cách chuẩn bị **một lần** dependencies/server, đường dữ liệu E1/E2/E2B/KLRD cũ cần có, lệnh `bash run_experiment3.sh` để **chạy training pilot**.
3. Các lệnh **riêng biệt** cho evaluate/merge, paraphrase opt-in và report/package; cách resume sau preemption.
4. Những giả định chưa xác nhận do máy local không có GPU, thông tin protocol/dataset nào còn thiếu; không nói đã PASS nếu chưa chạy.
5. Dự kiến nguồn chi phí chính và cách tối ưu **mà không thay KL loss hoặc verifier**. Không đưa yêu cầu chạy 30 diagnostics hoặc một chuỗi gate vòng vo.

**Kết quả khoa học mong muốn:** LoRA merged model bình thường sinh ra fingerprint detect được bằng `SphericalBlackBoxVerifier` trên held-out E2 prompts. Nếu đạt, high-fidelity GPT-5 mini paraphrase là phép thử bổ sung, không phải training objective hay lý do đưa PSS trở lại. Nếu không đạt, kết luận đúng là **student transfer chưa thành công**, không lấy teacher detection làm thay thế.
