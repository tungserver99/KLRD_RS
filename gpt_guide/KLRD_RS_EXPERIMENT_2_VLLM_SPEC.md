# KLRD-RS Experiment 2 — Chuyển Generation Backend sang vLLM

## 1. Phạm vi nhiệm vụ

Experiment 2 đã được triển khai. Hiện tại phần generation bằng Hugging Face quá chậm.

**Nhiệm vụ duy nhất của Codex là sửa code để chuyển generation backend sang custom vLLM, giữ nguyên toàn bộ Experiment 2 hiện có.**

### QUY ĐỊNH BẮT BUỘC VỀ MÔI TRƯỜNG

**MÁY LOCAL KHÔNG CÓ GPU. CODEX CHỈ ĐƯỢC VIẾT VÀ SỬA CODE.**

Tuyệt đối KHÔNG thực hiện trên máy local:

- Không chạy model, inference, generation hoặc evaluation.
- Không chạy GPU tests, smoke tests hoặc performance benchmarks.
- Không tải model weights hoặc datasets lớn.
- Không chạy các script thí nghiệm, dù chỉ với một vài samples.
- Không cài đặt lại CUDA, PyTorch, vLLM hoặc các dependencies nặng.
- Không tự động chạy các script vừa viết để kiểm tra.
- Không tạo GPU processes hoặc thực hiện các phép tính model lớn trên CPU.

Codex chỉ cần nghiên cứu source code, chỉnh sửa implementation, viết tests và chuẩn bị các câu lệnh kiểm thử.

**Tôi sẽ tự đưa code lên server có GPU và tự chạy toàn bộ tests, benchmarks và experiments.**

Không được tự ý thực hiện bất kỳ bước kiểm thử runtime nào trên máy local.

## 2. Hướng dẫn kỹ thuật tích hợp vLLM

Tham khảo Qwen2 implementation:

https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py

Và custom model registration:

https://docs.vllm.ai/en/stable/contributing/model/registration/

Nghiên cứu cơ chế `Qwen2Model.forward()` và final RMSNorm trong phiên bản vLLM mà repository dự kiến sử dụng.

Ở các phiên bản có fused residual addition + RMSNorm, code cuối decoder có thể có dạng:

`hidden_states, _ = self.norm(hidden_states, residual)`

KLRD-RS phải áp dụng **exact latent rotation lên residual state ngay trước final RMSNorm**.

Về mặt toán học, trình tự là:

1. Xác định pre-normalization residual state, bao gồm residual addition chính xác.
2. Áp dụng `R_K` lên state đó.
3. Chạy original final RMSNorm, không cộng residual lần thứ hai.
4. Chạy original LM head.
5. Trả logits cho vLLM sampler.

Lưu ý: đây là mô tả vị trí can thiệp, không phải giả định rằng mọi phiên bản vLLM đều có API giống nhau.

Codex phải đọc implementation tương ứng và xử lý đúng fused/non-fused paths.

## 3. Custom Qwen2 model

Ưu tiên triển khai trong repository, không sửa trực tiếp thư viện vLLM đang cài đặt.

Có thể dùng custom `Qwen2Model` / `Qwen2ForCausalLM` và đăng ký bằng `ModelRegistry.register_model()` theo API của phiên bản vLLM mục tiêu.

Cần bảo đảm:

- Load được model weights gốc.
- Áp dụng exact rotation ở đúng vị trí.
- Tương thích với prefill và incremental decoding.
- Giữ nguyên vLLM attention, KV cache và continuous batching.
- Hỗ trợ một fixed owner key cho mỗi generation run.

Không dùng approximate logit bias thay cho latent rotation.

Nếu không thể xác minh phiên bản vLLM từ repository, hãy viết integration cho một phiên bản mục tiêu được khai báo rõ và ghi lại yêu cầu tương thích; không được tự chạy cài đặt vLLM để dò thử trên máy local.

## 4. Giữ nguyên cấu hình KLRD-RS

Tái sử dụng frozen artifacts của Experiment 1:

- Qwen2.5-3B-Instruct và revision đã cố định.
- Rotation basis và Fisher whitening.
- Global λ = 0.09271215438842773.
- Secret keys, seeds và sampling settings.
- Exact teacher distribution.
- Spherical verifier và evaluation protocol.

Không thay đổi phương pháp để tối ưu tốc độ.

Không dùng FP64 Linear trong generation loop. Sử dụng BF16/FP32 phù hợp, nhưng phải giữ khả năng kiểm tra numerical parity trên server.

Đảm bảo temperature chỉ được áp dụng một lần, không vừa chia logits trong custom model vừa chia lần nữa trong sampler.

## 5. Chỉ sửa generation backend hiện tại

Tận dụng code Experiment 2 đã có.

Giữ nguyên:

- Prompt manifests.
- Owner-key assignments.
- Clean/fingerprint generation tasks.
- Output files và schema.
- Resume/checkpoint mechanism.
- Verification, evaluation và metrics.
- Toàn bộ experimental protocol.

Chỉ sửa những phần trực tiếp cần thiết để generation sử dụng vLLM.

Không viết lại Experiment 2 từ đầu.

Không sử dụng Hugging Face generation làm fallback.

## 6. Viết tests nhưng KHÔNG CHẠY

Codex cần viết các tests hoặc scripts để tôi tự chạy trên server GPU sau này:

1. Clean-model parity khi rotation bị vô hiệu hóa.
2. Exact rotation parity với công thức Experiment 1.
3. Correct residual addition và final RMSNorm.
4. Prefill/incremental decoding correctness.
5. KV-cache và batched-generation correctness.
6. Rotated logits và sampling-distribution correctness.
7. Benchmark throughput và GPU memory.

**Chỉ viết code tests, không thực thi bất kỳ test nào trên máy local.**

Không được báo cáo PASS khi chưa có kết quả thực thi trên server.

Chuẩn bị sẵn các lệnh chạy tương ứng, để tôi có thể tự copy và thực hiện trên server.

## 7. Giới hạn cuối cùng

Codex chỉ được:

- Đọc source code và cấu hình hiện có.
- Chỉnh sửa generation backend.
- Viết custom vLLM integration.
- Viết tests và benchmark scripts.
- Viết hướng dẫn/lệnh chạy trên server.

Codex KHÔNG được:

- Chạy bất kỳ model hoặc experiment nào trên máy local.
- Tự tải weights, sinh responses hoặc benchmark.
- Cài đặt dependencies nặng để kiểm thử.
- Chuyển sang LoRA, PSS hoặc Experiment 3.
- Thay đổi phương pháp hay quy mô Experiment 2.
- Tự đóng gói kết quả thí nghiệm.

**Kết quả mong muốn: Code Experiment 2 hiện tại được sửa để sử dụng exact KLRD-RS teacher generation bằng vLLM, cùng các scripts kiểm thử có thể chạy trên server GPU. Codex dừng ở mức hoàn thiện code; mọi hoạt động thực thi sẽ do tôi tự thực hiện trên server.**