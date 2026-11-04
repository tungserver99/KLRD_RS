# KLRD-RS Experiment 3 — Full LoRA Training, 1024-Token Evaluation, ARC-C Utility & Paraphrase

## 1. Mục tiêu

Tiếp tục Experiment 3 của KLRD-RS, dựa trực tiếp trên implementation và các kết quả pilot đã hoàn thành.

Experiment 3 pilot hiện tại:

- Base model: Qwen2.5-3B-Instruct.
- Exact KLRD-RS Teacher với owner key 0.
- LoRA Student được huấn luyện bằng full-vocabulary KL distillation.
- Training: 2.000 sequences.
- Validation: 200 sequences.
- LoRA rank 16, alpha 32.
- 2 epochs.
- Evaluation: `max_new_tokens=512`.

Kết quả pilot cho thấy LoRA Student đã học được fingerprint trong weights. Fingerprint vẫn phát hiện được sau high-fidelity paraphrase tại Q=100 với ngưỡng \(\alpha=0.01\), nhưng tín hiệu yếu hơn Exact Teacher.

Ngoài ra, khoảng 67% responses của Clean, Teacher và Student bị cắt tại giới hạn 512 tokens.

### Những thay đổi cần thực hiện

1. **Mở rộng LoRA training lên 10.000 training sequences và 1.000 validation sequences.**
2. **Tăng evaluation generation length từ 512 lên 1024 tokens.**
3. **So sánh Pilot LoRA 2k với Full LoRA 10k trên cùng điều kiện 512 tokens.**
4. **Đánh giá fingerprint trên Clean, Exact Teacher và Full LoRA Student tại 1024 tokens.**
5. **Bổ sung ARC-Challenge (ARC-C) vào utility evaluation, tái sử dụng code ARC-C bên KLRD.**
6. **Chuẩn bị và hỗ trợ chạy high-fidelity paraphrase trên Full LoRA Student.**

Không thay đổi phương pháp KLRD-RS, không triển khai PSS và không thiết kế fingerprint mechanism mới.

---

## 2. Rà soát code trước khi thực hiện

Đọc toàn bộ implementation Experiment 3 hiện có, bao gồm các module liên quan:

- `main_exp3.py`
- `klrd_rs/e3/config.py`
- `klrd_rs/e3/generation.py`
- `klrd_rs/e3/evaluate.py`
- `klrd_rs/e3/paraphrase.py`
- `klrd_rs/e3/provenance.py`
- Các module training, dataset preparation, checkpoint và LoRA merge.
- Các module vLLM backend và verifier của Experiment 2.

Đồng thời, **tìm lại implementation ARC-C trong project KLRD trước đây** để reuse.

Không tự viết lại các phần đã có nếu có thể dùng lại an toàn.

Phải kiểm tra sự tương thích giữa model loading, checkpoint format, inference backend và evaluation scripts trước khi tích hợp.

Giữ nguyên toàn bộ kết quả cũ.

---

# PHASE A — FULL LoRA TRAINING

## 3. Training configuration

Triển khai một full-training run mới với cấu hình:

| Parameter | Pilot | Full training |
|---|---|---|
| Base model | Qwen2.5-3B-Instruct | Giữ nguyên |
| Owner key | 0 | 0 |
| Training samples | 2.000 | **10.000** |
| Validation samples | 200 | **1.000** |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| Learning rate | 1e-5 | 1e-5 |
| Effective batch size | 16 | 16 |
| Epochs | 2 | 2 |
| Optimizer steps | 250 | Khoảng 1.250 |
| Objective | Full-vocabulary forward KL | Giữ nguyên |

Giữ nguyên những thiết lập khác của pilot:

- Pinned base model revision.
- LoRA target modules.
- Optimizer và learning-rate scheduler.
- Training sequence length.
- Training temperature policy.
- Loss masking.
- EOS handling.
- Teacher/student precision.
- Dataset preprocessing.
- Checkpoint/resume behavior.

Không điều chỉnh thêm hyperparameters nếu không có lý do kỹ thuật bắt buộc.

### 3.1. Dataset

Sử dụng:

- 10.000 training sequences.
- 1.000 validation sequences.

Đảm bảo:

- Train và validation không overlap.
- Không overlap với frozen evaluation prompts.
- Giữ nguyên dataset source và selection policy của pilot.
- Không thay đổi domain hoặc sampling distribution chỉ để đạt đủ số lượng.
- Có manifest và hashes để kiểm tra tính nhất quán.

10.000 là tổng số training samples của full run, không phải 10.000 samples cộng thêm vào pilot.

### 3.2. Exact teacher

Giữ nguyên KLRD-RS Exact Teacher:

\[
a_k=\lambda Lk
\]

Trong đó \(L\), \(\lambda\), P/Q và key bank đều lấy từ frozen Experiment 1 artifacts.

Dùng owner key 0.

Teacher phải sử dụng exact latent rotation tại đúng vị trí trong forward pass, theo implementation đã được kiểm tra của Experiment 2/3.

Không thay bằng:

- Tangent approximation.
- Logits bias.
- SFT trên generated responses.
- Preference learning.
- Một cơ chế fingerprint khác.

### 3.3. Distillation objective

Giữ nguyên full-vocabulary KL distillation:

\[
\mathcal L_{\mathrm{KL}}
=
\mathbb E_c
\left[
D_{\mathrm{KL}}\left(q_k(\cdot\mid c)\Vert p_\theta(\cdot\mid c)\right)
\right]
\]

Trong đó:

- \(q_k\): frozen Exact Teacher.
- \(p_\theta\): LoRA Student.
- Chỉ cập nhật LoRA parameters.
- Giữ nguyên temperature, masking và normalization policy của pilot.

Không chuyển thành supervised learning trên teacher-generated text.

### 3.4. Khởi tạo training

**Full LoRA training phải bắt đầu từ clean base model với adapter mới.**

Không tiếp tục train từ pilot checkpoint 2k.

Mục tiêu là so sánh công bằng:

- Pilot: 2.000 samples × 2 epochs.
- Full: 10.000 samples × 2 epochs.

Giữ fixed owner key và phương pháp.

### 3.5. Checkpoints

Lưu:

- Training KL.
- Validation KL.
- Best-validation checkpoint.
- Final checkpoint.
- Merged LoRA model.
- Training configuration.
- Dataset manifests.
- Checkpoint hashes.
- Frozen artifact identifiers.

Có checkpoint/resume để tiếp tục khi PBS job bị ngắt.

Không chọn checkpoint theo test TPR hoặc paraphrase performance.

---

# PHASE B — ORIGINAL FINGERPRINT EVALUATION

## 4. Evaluation tại 512 tokens: Pilot vs Full

Đây là comparison bắt buộc để kiểm tra việc tăng training data.

Đánh giá:

| Model | Training data | Max new tokens |
|---|---|---|
| Pilot LoRA Student | 2k | 512 |
| Full LoRA Student | 10k | 512 |

Dùng cùng:

- Frozen evaluation prompts.
- Owner key 0.
- Sampling configuration.
- Prompt grouping.
- Spherical verifier.
- Significance thresholds.

Có thể reuse pilot responses và scores nếu provenance hợp lệ.

Không cần generate lại pilot khi đã có dữ liệu đúng.

Báo cáo:

- Validation KL của hai checkpoints.
- TPR theo query budgets.
- Correct-key cosine.
- Wrong-key results.
- Utility/generation statistics.
- Chênh lệch giữa pilot và full training.

**Mục đích: xác định fingerprint có mạnh hơn nhờ tăng training data hay không.**

Không dùng kết quả pilot 512 vs full 1024 để kết luận hiệu quả training, vì hai cấu hình generation khác nhau.

---

## 5. Evaluation tại 1024 tokens

Tạo một evaluation run mới cho ba model.

### Model A — Clean

Qwen2.5-3B-Instruct gốc.

Không fingerprint, rotation hoặc LoRA.

### Model B — Exact Teacher

Exact KLRD-RS Teacher với owner key 0.

Sử dụng đúng frozen artifacts và exact rotation implementation.

### Model C — Full LoRA Student

Merged checkpoint của full training 10k.

Inference thông thường, hoàn toàn weight-resident:

- Không sử dụng secret key tại runtime.
- Không rotation hooks.
- Không fingerprint logits processor.
- Không custom fingerprint sampling mechanism.

### 5.1. Evaluation prompts

Dùng đúng 1.000 frozen Experiment 3 group0 prompt IDs.

Giữ nguyên:

- Prompt text.
- Chat template.
- Prompt ordering.
- Grouping policy.
- Random seed policy.

Không chọn lại prompts dựa trên TPR.

### 5.2. Generation configuration

Cho cả Clean, Teacher và Student:

- `max_new_tokens=1024`
- Temperature = 0.8.
- Top-p = 1.0.
- Không giới hạn top-k.
- Một response/prompt.
- Native EOS handling.
- Giữ các sampling settings còn lại của pilot.

Không sửa thêm generation configuration.

### 5.3. Context length

Kiểm tra `max_model_len`.

Phải đảm bảo khả năng chứa prompt tokens và tối đa 1024 generated tokens, tính cả chat template và special tokens.

Nếu context capacity hiện tại không đủ, thay đổi trong evaluation run mới.

Không được:

- Truncate prompts.
- Loại bỏ những prompts dài.
- Cắt responses thủ công.
- Thay đổi evaluation population.

Giữ nguyên exact-teacher precision và inference semantics.

---

# PHASE C — FINGERPRINT VERIFICATION

## 6. Spherical verification

Reuse verifier hiện tại của KLRD-RS.

Không thay đổi verifier hoặc thêm empirical calibration.

Verification phải dựa trên black-box responses:

1. Nhận prompt và response text.
2. Tokenize bằng canonical tokenizer.
3. Teacher-force trên frozen clean reference model.
4. Tính Fisher-whitened tangent score.
5. Aggregate score vectors theo nhóm.
6. Tính cosine và analytical spherical p-value.

Không dùng generation-time hidden states để verify.

### 6.1. Query budgets

Đánh giá:

\[
Q\in\{20,50,100,1000\}
\]

Với 1.000 prompts:

| Q | Số nhóm không chồng lặp |
|---|---:|
| 20 | 50 |
| 50 | 20 |
| 100 | 10 |
| 1000 | 1 |

Dùng cùng group IDs cho các model.

### 6.2. Thresholds

\[
\alpha\in\{0.01,10^{-4}\}
\]

Giữ nguyên analytical random-key null.

Không dùng clean-response calibration để đặt threshold.

Phải phân biệt rõ:

- Nominal FPR từ analytical null.
- Empirical/observed clean false-positive rate.

### 6.3. Metrics

Xuất:

- Correct-key TPR.
- Observed clean FPR.
- Wrong-key detection results (20 controls).
- Correct-key cosine.
- Analytical p-values.
- Closed-set identification nếu đã có.
- Per-group decisions.
- Group-level sample counts.

Không trộn responses 512 và 1024 tokens trong cùng phép tính TPR.

---

# PHASE D — UTILITY EVALUATION

## 7. Utility evaluation tổng thể

So sánh trực tiếp:

**Clean vs Exact Teacher vs Full LoRA Student**

trên cùng evaluation protocol.

Utility gồm hai phần:

1. Generation statistics và kiểm tra chất lượng responses.
2. **ARC-Challenge (ARC-C) benchmark.**

---

## 8. Generation utility

Dùng 1.000 frozen evaluation prompts.

Báo cáo:

- Mean generated tokens.
- Median generated tokens.
- EOS termination rate.
- Max-length truncation rate.
- Empty response rate.
- Severe repetition rate.
- Text-token roundtrip consistency.

So sánh tại `max_new_tokens=1024`:

| Metric | Clean | Teacher | Full Student |
|---|---|---|---|
| Mean generated tokens | | | |
| Median generated tokens | | | |
| EOS termination rate | | | |
| Truncation rate | | | |
| Severe repetition rate | | | |
| Empty response rate | | | |

Đồng thời so sánh truncation ở 512 và 1024 tokens.

**Chú ý:** Muốn kết luận tác động của generation length thì phải so sánh trên cùng model checkpoint, không được nhầm với ảnh hưởng của training dataset size.

### Response examples

Tạo private comparison file chứa ít nhất 10 frozen examples:

`Prompt / Clean / Exact Teacher / Full LoRA Student`

Chọn IDs trước khi xem kết quả, không cherry-pick.

Dùng để đánh giá thủ công coherence, factual correctness, reasoning và formatting.

Không công khai raw private prompts/responses trong review ZIP.

---

## 9. ARC-Challenge (ARC-C) — BỔ SUNG BẮT BUỘC

**Tái sử dụng trực tiếp code ARC-C utility evaluation từ project KLRD trước đây.**

Không cần xây dựng một ARC-C benchmark pipeline mới.

### 9.1. Rà soát code KLRD

Tìm module/script ARC-C hiện có trong KLRD.

Reuse:

- Dataset loader.
- Dataset version và test split.
- Prompt construction.
- Answer-choice formatting.
- Few-shot/zero-shot settings.
- Answer extraction hoặc choice-scoring implementation.
- Metric calculation.
- Evaluation batching.
- Result serialization.

**Giữ nguyên evaluation protocol của KLRD để kết quả có thể so sánh trực tiếp.**

Nếu implementation KLRD dùng log-likelihood scoring trên answer choices, tiếp tục sử dụng cách đó. Nếu dùng answer generation, giữ đúng original generation/scoring protocol.

Không tự chuyển từ phương pháp này sang phương pháp kia.

Không thay dataset, prompt template, số shot hoặc cách xử lý đáp án.

### 9.2. Models cần đánh giá

Tối thiểu chạy ARC-C cho:

**A. Clean Qwen2.5-3B-Instruct**

- Original base model.
- Không fingerprint.

**B. Exact KLRD-RS Teacher**

- Same base model.
- Owner key 0.
- Exact rotation.
- Không sử dụng approximation.

**C. Full LoRA Student (10k)**

- Merged LoRA model.
- Ordinary inference.
- Không runtime key injection.

Nếu khả thi với code hiện tại, chạy thêm:

**D. Pilot LoRA Student (2k)**

Để so sánh trực tiếp pilot và full student trên cùng ARC-C protocol.

### 9.3. Những điều không được thay đổi

- Không chỉnh ARC-C dataset để phù hợp student.
- Không đổi few-shot settings.
- Không sửa prompt cho từng model.
- Không thay scoring để làm kết quả tốt hơn.
- Không dùng ARC-C test accuracy để chọn checkpoint.
- Không lẫn ARC-C với 1.000 frozen fingerprint prompts.

ARC-C là một benchmark utility riêng biệt.

**Không cần áp dụng `max_new_tokens=1024` một cách máy móc cho ARC-C.** Giữ đúng inference/scoring settings của ARC-C code trong KLRD. Thay đổi 1024 tokens chỉ áp dụng cho free-form generation evaluation và paraphrase pipeline.

### 9.4. Kết quả ARC-C cần xuất

Bảng:

| Model | ARC-C Accuracy (%) | Δ vs Clean (percentage points) |
|---|---:|---:|
| Clean | | 0 |
| Exact Teacher | | |
| Pilot LoRA 2k | | |
| Full LoRA 10k | | |

Đối với mỗi model, lưu:

- Number of evaluated examples.
- Correct count.
- Total count.
- Accuracy.
- Evaluation configuration.
- Model/checkpoint identifier.
- Runtime nếu có.

Đảm bảo tất cả các model dùng chính xác cùng ARC-C test examples và protocol.

### 9.5. Kết luận utility cần kiểm tra

Trả lời:

1. Exact Teacher có làm giảm ARC-C accuracy so với Clean không?
2. Full LoRA Student có giữ được ARC-C accuracy gần Clean không?
3. Full LoRA Student có giảm accuracy so với Pilot 2k không?
4. Việc tăng training data có cải thiện fingerprint mà vẫn duy trì ARC-C utility không?

Không đưa ra kết luận trước khi có kết quả.

**Ưu tiên giữ utility của student sát Clean**, nhưng không dùng ARC-C để tối ưu fingerprint hoặc chọn checkpoint.

---

# PHASE E — HIGH-FIDELITY PARAPHRASE

## 10. Paraphrase evaluation

Sau khi full training và original evaluation hoàn thành, hỗ trợ high-fidelity paraphrase evaluation.

Dùng:

- 400 frozen evaluation prompt IDs.
- GPT-5 mini version của Experiment 2B run_002.
- Exact high-fidelity paraphrase prompt của run_002.
- Cùng text-preservation checks và API policies.

Không sửa hoặc phát triển prompt mới.

### 10.1. Paraphrase conditions

Đánh giá:

- Clean.
- Exact Teacher.
- Full LoRA Student.

Mỗi model có original responses được generate ở 1024 tokens.

Paraphrase phải thực hiện trên đúng original responses của run này.

**Không reuse paraphrase texts từ các response 512 tokens để thay cho bản 1024 tokens.**

Có thể reuse prompt IDs, key material và verifier configuration.

Không gọi lại API cho các responses đã paraphrase thành công trong cùng run nếu có cache hợp lệ.

### 10.2. Metrics

Đánh giá:

\[
Q\in\{20,50,100\}
\]

Với 400 responses:

| Q | Số nhóm |
|---|---:|
| 20 | 20 |
| 50 | 8 |
| 100 | 4 |

Thresholds:

\[
\alpha\in\{0.01,10^{-4}\}
\]

Báo cáo:

- Original TPR.
- Paraphrased TPR.
- Clean false-positive rate.
- Wrong-key controls.
- Correct-key cosine trước và sau paraphrase.
- Mean/median cosine.
- Length ratio.
- Word similarity.
- Numeric-literal preservation.
- Structural changes.

Đặc biệt quan tâm đến hiệu quả tại **TPR dưới nominal 1% FPR**, nhất là Q=20, 50 và 100.

### 10.3. API execution

Paraphrase phải là một phase riêng.

Không tự động gọi GPT-5 mini trong training hoặc original evaluation.

Chỉ gọi API khi người dùng chạy lệnh paraphrase rõ ràng.

Kiểm tra giới hạn input/output tokens vì responses 1024 tokens dài hơn pilot 512 tokens.

Lưu provenance và API usage.

### 10.4. Không bổ sung phương pháp mới

Không thêm:

- PSS.
- Paraphrase-specific loss.
- Auxiliary semantic objectives.
- Một verifier thay thế.

Mục tiêu hiện tại chỉ là xem full-training LoRA có giữ fingerprint sau high-fidelity paraphrase tốt hơn pilot hay không.

---

# PHASE F — IMPLEMENTATION, PBS VÀ OUTPUTS

## 11. Output organization

Tạo full experiment run mới, tách biệt toàn bộ runs đã có.

Ví dụ:

```text
experiment3_full/
  run_001/
    config/
    data_manifest/
    checkpoints/
    merged_student/
    training_metrics/
    eval512/
      generation/
      scores/
      report/
    eval1024/
      generation/
      scores/
      utility/
      paraphrase/
      report/
    arc_c/
      clean/
      teacher/
      pilot_student/
      full_student/
      comparison.csv
    private/
      examples.md
    report/
      full_experiment_report.md
```

Có thể điều chỉnh cấu trúc nếu repository đã có convention tốt hơn.

Không ghi đè:

- Experiment 2 runs.
- Experiment 2B paraphrase runs.
- Experiment 3 pilot run.
- Existing frozen artifacts.
- Existing model checkpoints.

## 12. CLI và execution workflow

Ưu tiên mở rộng `main_exp3.py` hoặc tạo một entrypoint độc lập phù hợp với architecture hiện tại.

Cần hỗ trợ các phase:

1. Full LoRA training.
2. Evaluation 512.
3. Evaluation 1024.
4. ARC-C utility.
5. Paraphrase 1024.
6. Final report/export.

Ví dụ giao diện mong muốn:

```bash
# Phase A: Full LoRA training
python main_exp3.py \
  --mode train \
  --output experiment3_full/run_001
```

```bash
# Phase B: Evaluation 512
python main_exp3.py \
  --mode evaluate \
  --output experiment3_full/run_001 \
  --max-new-tokens 512
```

```bash
# Phase B/C: Evaluation 1024
python main_exp3.py \
  --mode evaluate \
  --output experiment3_full/run_001 \
  --max-new-tokens 1024
```

```bash
# Phase D: ARC-C utility
python main_exp3.py \
  --mode arc-c \
  --output experiment3_full/run_001
```

```bash
# Phase E: High-fidelity paraphrase
python main_exp3.py \
  --mode paraphrase \
  --output experiment3_full/run_001 \
  --max-new-tokens 1024
```

Đây là **CLI mong muốn để Codex triển khai**, không phải khẳng định các options đã tồn tại.

Không tạo những entrypoints hoặc flags không cần thiết nếu code hiện có đã hỗ trợ tương đương.

### PBS requirements

Chuẩn bị hướng dẫn/script để chạy từng phase trên GPU server hoặc CETUS PBS.

Yêu cầu:

- Mỗi phase chạy độc lập.
- Checkpoint/resume.
- Không lặp lại các batches đã hoàn thành.
- Có logging và error reporting.
- Không tự động chạy phase tiếp theo.
- Không tự động gọi external API.
- Không giữ đồng thời nhiều model nếu không cần thiết.
- Phù hợp với A100 40GB.

---

## 13. Final report

Tạo một báo cáo tổng kết và review ZIP.

### Table 1 — LoRA training comparison

Pilot 2k vs Full 10k:

- Train/validation KL.
- Best checkpoint.
- Optimizer steps.
- Training runtime.
- Original fingerprint TPR tại 512 tokens.
- Correct-key cosine.

### Table 2 — Fingerprint verification at 1024 tokens

Clean / Exact Teacher / Full LoRA Student:

- TPR theo Q và α.
- Clean FPR.
- Wrong-key controls.
- Correct-key cosine.
- Closed-set identification nếu có.

### Table 3 — High-fidelity paraphrase

Clean / Exact Teacher / Full LoRA Student:

- Original vs paraphrased TPR.
- Q=20/50/100.
- α=0.01 và 10⁻⁴.
- Cosine comparison.
- Paraphrase preservation metrics.

### Table 4 — Utility: ARC-C

Clean / Exact Teacher / Pilot LoRA / Full LoRA:

- ARC-C accuracy.
- Accuracy change versus Clean.
- Correct/total counts.

### Table 5 — Generation utility

Clean / Exact Teacher / Full LoRA:

- Generated length.
- EOS rate.
- Truncation rate.
- Severe repetition.
- Empty response rate.

### Table 6 — 512 vs 1024 tokens

So sánh generation length trên cùng checkpoint:

- Truncation.
- Original TPR.
- Correct-key cosine.
- Generation throughput.

### Discussion

Báo cáo cần trả lời:

1. Full training 10k có giảm validation KL so với pilot 2k không?
2. Full student có fingerprint mạnh hơn pilot không?
3. TPR tại Q=20/50/100 có cải thiện không?
4. Full student có giữ fingerprint sau paraphrase tốt hơn không?
5. ARC-C accuracy của student có giảm đáng kể so với Clean không?
6. Generation length 1024 có giảm truncation đáng kể không?
7. Student có suy giảm generation utility so với Clean không?

Không suy đoán hoặc tuyên bố phương pháp thành công khi chưa có evidence.

Review ZIP cần chứa code/config, metrics, tables, provenance và logs cần thiết, nhưng không chứa model weights, secret keys hoặc raw private prompts/responses.

---

## 14. Những hạn chế bắt buộc đối với Codex

**Máy local không có GPU. Codex chỉ có nhiệm vụ đọc và chỉnh sửa code.**

Không được:

- Chạy training.
- Chạy model inference.
- Chạy numerical tests hoặc benchmarks.
- Download model weights.
- Cài GPU dependencies.
- Gọi GPT-5 mini API.
- Chạy PBS jobs từ local.

Tôi sẽ tự chạy toàn bộ experiments trên GPU server.

Codex cần bàn giao:

- Source code hoàn chỉnh.
- Training/evaluation/PBS scripts.
- Hướng dẫn chạy từng phase.
- Danh sách files đã sửa.
- Mô tả các assumptions và limitations.
- Cách tái sử dụng code ARC-C từ KLRD.
- Cách kiểm tra provenance và resume.

Không sửa phương pháp KLRD-RS.

---

## 15. Thứ tự triển khai bắt buộc

**Step 1 — Code review**

Rà soát Experiment 3 và ARC-C code từ KLRD.

**Step 2 — Full LoRA training**

Implement 10.000 train / 1.000 validation sequences, training từ clean base, có checkpoints và resume.

**Step 3 — 512-token evaluation**

So sánh Pilot 2k vs Full 10k, giữ cùng generation settings.

**Step 4 — 1024-token evaluation**

Clean vs Exact Teacher vs Full Student, trên cùng 1.000 frozen prompts.

**Step 5 — Verification**

Giữ nguyên spherical verifier, analytical p-values và các query budgets.

**Step 6 — ARC-C utility**

Reuse original KLRD ARC-C evaluation code, áp dụng cho Clean, Teacher và LoRA checkpoints; không đổi scoring protocol.

**Step 7 — Generation utility**

So sánh response lengths, truncation, EOS, repetition và private examples.

**Step 8 — Paraphrase**

Chuẩn bị optional high-fidelity paraphrase evaluation; không gọi API.

**Step 9 — Reporting**

Chuẩn bị bảng kết quả, logs, review ZIP và PBS run instructions.

## Yêu cầu cuối cùng

Hoàn thành implementation để tôi có thể:

- Train Full KLRD-RS LoRA Student bằng 10k training sequences.
- So sánh trực tiếp với Pilot 2k.
- Đánh giá fingerprint ở 512 và 1024 tokens.
- Đánh giá **ARC-C utility bằng code KLRD sẵn có**.
- Đánh giá high-fidelity paraphrase robustness.
- Xuất toàn bộ bảng kết quả phục vụ paper.

**Không thay đổi fingerprint method, không thay verifier, không triển khai PSS và không tự thực thi experiments trên local.**