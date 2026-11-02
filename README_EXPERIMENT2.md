# Experiment 2 — exact teacher free generation

**Runtime hiện tại: vLLM 0.32.0 / V1, FP32 / TRITON_ATTN; output `experiment2_rs/run_012/`.**

**Mặc định launcher: pilot 9.000 primary responses, không chạy full 47.000.**
Dùng group0 đã chốt: 8 owner keys × 1.000 prompts = 8.000 fingerprint responses,
cộng 1.000 clean responses; không sinh stress keys23/59. Mỗi key có 10 nhóm
Q=100 không giao nhau và 1 nhóm Q=1000; Q=100 vẫn lồng trong Q=1000.
Các lượt smoke hiện có (20 prompts × clean/key0 = 40 responses) là bổ sung,
không tính vào 9.000 primary responses. Không thêm numerical gates, không đổi
vLLM, precision, verifier, KL audit, key/seed/λ hoặc dữ liệu frozen.

Dừng tiến trình full hiện tại trước khi chạy lại launcher. Lượt chạy vẫn dùng
`experiment2_rs/run_012/`; các batch group0 đã commit và score cache hợp lệ được
dùng lại nếu code/runtime/config/output không đổi. Thay đổi này chỉ chạm launcher,
tài liệu và test contract; các file Python tham gia job identity được giữ nguyên.
Không xóa dữ liệu cũ. Nếu run_012 đã có nhóm1–4/stress, analyzer hiện tại vẫn tổng
hợp dữ liệu đã lưu đó; báo cáo không phải một tập pilot group0 thuần túy.

Pilot kết thúc với nhãn `PILOT_ONLY`; `complete=False`/exit code1 thể hiện chưa
hoàn tất full study, không tự động kết luận phương pháp thất bại. Đọc số lượng
nhóm thực có trong báo cáo; mẫu số kỳ vọng full vẫn giữ nguyên. Pilot chưa cho
phép kết luận FPR mạnh hoặc tự động chuyển sang LoRA.

Khi chủ động mở rộng full, từ cùng môi trường/thư mục/output chạy:

```python
!python main_exp2.py
```

Lệnh full này tái sử dụng các batch pilot hợp lệ; `!bash run_experiment2.sh` vẫn
luôn chọn pilot. Không tự mở rộng full sau pilot.

Lệnh notebook giữ nguyên:

```python
!bash run_experiment2.sh
```

File `run_experiment2.sh` vẫn chỉ chứa `python main_exp2.py --pilot-only`. Chuẩn bị environment
riêng trên GPU server bằng `requirements-vllm.txt`; Python >=3.11,<3.15 và
Transformers >=5.16.1,<5.20.0. Khởi chạy notebook từ environment mới để `python`
trong PATH trỏ đúng môi trường. Không nâng package trong environment đang chạy.
Xem [hướng dẫn vLLM](README_VLLM_EXPERIMENT2.md) để cài đặt và kiểm tra tùy chọn.

Generation yêu cầu CUDA Graph decode, cập nhật rotation buffers tại chỗ giữa
các batch cố định key. Optional probes dùng engine eager riêng; worker cũ phải
thoát trước khi engine khác khởi tạo. Finite-logits vẫn tích lũy trên GPU và đọc
một lần/batch. Prefix caching tắt; production bật chunked prefill. Các thiết lập
engine mới nằm trong runtime identity, nên không resume lẫn kết quả 0.10.2.
Run_001 đến run_011 được bảo vệ; không sửa hash/checkpoint hoặc nhập observations
cũ vào run_012. Prompt/cohort/key/seed/λ/sampling/EOS và protocol E2 giữ nguyên.

Verifier giữ phần tối ưu đã có: chunk execution tối thiểu64 token, FP64 Linear
và tổng trên GPU, chuyển về CPU một lần/batch; NumPy key projection, spherical
verifier và logic KL giữ nguyên. `score_chunk=4` vẫn nằm trong frozen protocol.
Numerical diagnostics mặc định tắt và không blocking; artifact/runtime/NaN/Inf
vẫn báo lỗi. `sampler_valid` xác nhận runtime configuration, không tuyên bố parity.

**Trạng thái: đã sửa source, chưa chạy test/inference/benchmark trên GPU server.**
Không có cam kết speedup đã đo. Đồng bộ `main_exp2.py`, toàn bộ `klrd_rs/e2/`
(kể cả `vllm_runtime.py`), requirements, READMEs và tool diagnostic cùng phiên bản.

Chạy từ thư mục dự án với Python/CUDA environment của bạn:

```bash
bash run_experiment2.sh
```

File `.sh` chỉ chứa `python main_exp2.py --pilot-only`, không cài env, không kích hoạt env,
không chứa đường dẫn Python hoặc yêu cầu file config thủ công.

Tiến độ/ETA hiện trực tiếp khi chạy file sh, không cần monitor riêng. Chạy lại
cùng lệnh để resume run_012 khi code/runtime/config/artifact không thay đổi.
Receipt `runner_progress_resume.json` chỉ dành cho lần sửa logging lịch sử của
run_010, không cho phép resume run_010/011 với migration này. Không chỉnh/xóa
identity hay checkpoint để ép resume và không dừng lượt chạy cũ để nâng package.
Nếu dùng monitor read-only riêng cho lượt mới:
`python tools/watch_exp2_progress.py --output experiment2_rs/run_012`.
Ví dụ định dạng (số minh họa): `E2 progress: 125/5,875 batches (2.13%) |
responses 1,000/47,000 | resumed 120 | ETA ~08:20:00`. ETA xuất hiện sau batch
mới đầu tiên, dựa trên tối đa 20 batch mới gần nhất; không tính thời gian tổng
hợp báo cáo. Có thông báo phase khi tổng hợp pilot/cuối. Tiến độ tính theo scope
đã chọn nếu dùng owner/group/chunk/pilot flags. Chưa chạy kiểm thử local.

Giữ toàn bộ code mới, **`e2_inputs_v2/`**, `inputs/calibration_manifest.json`, và
artifact public của **`experiment1_rs_continuation/`**. Runtime không cần truy cập
thư mục KLRD bên cạnh: exclusion index đã được chuẩn bị từ lịch sử KLRD và hai
selection E1. Không xóa/đổi frozen E1 artifacts. Dependencies bổ sung là `datasets`
và `filelock` trong requirements.txt; bạn tự chuẩn bị môi trường server như trước.

## Đường chạy chính

Sampler generation dùng **custom vLLM FP32, batched incremental KV cache**,
batch8 cố định. Không FP64 Linear và không fallback HF generation. HF chỉ giữ
vai trò verifier và replay đo KL theo native tokens, với precision cũ.
Chỉ readout cuối được thay bằng exact finite rotation của E1. Không replay toàn
prefix mỗi bước sinh, không dùng tangent approximation để sinh, không rerank,
không top-k/top-p filtering, không penalty và không áp dụng temperature hai lần.
Custom Qwen2 áp dụng rotation sau residual addition và trước final RMSNorm,
không dùng logit bias gần đúng. Runtime parity/throughput cần bạn xác nhận trên server.

Lần chạy mới sẽ tự động:

1. Kiểm tra SHA256 từ receipt đã khóa với ZIP E1, model/tokenizer revision,
   P/Q/L và λ=0.09271215438842773. Tái dựng đúng 64 key E1 và đối chiếu mọi hash.
   Không PCA/QR/Fisher fit lại, không chọn key mới hoặc đổi λ.
2. Chọn 5000 prompt mới, deterministic, loại IDs và exact/normalized text hashes
   từ historical KLRD manifests. Ưu tiên official test_gen ở dataset revision E1;
   thiếu thì dùng train_gen chưa dùng và ghi rõ mixed source. Prompt16–512 tokens.
   Không đủ5000 thì dừng `INSUFFICIENT_ELIGIBLE_FRESH_PROMPTS`, không giảm thiết kế.
3. Freeze nhóm trước khi nhìn output: năm nhóm Q1000, mỗi nhóm chia mười Q100.
   Mỗi owner có đúng5/50 disjoint trials. Sample quality100/example10/native-control20
   prompt IDs cũng được định trước. Nguồn exclusion, hash và split counts có trong
   provenance. Không có model-scored/semantic filtering.
4. Chạy gate trên **ít nhất10 actual Qwen prefill +20 KV decode contexts**, gồm key2,
   zero rotation, full native rotated readout, normalization, EOS-sensitive token,
   full-prefix vs KV, seeded reproduction và mô phỏng interruption sau commit.
   Max/p99 log-prob errors phải <=0.0002/0.00005. TF32 tắt; không có BF16 fallback.
   Gate fail thì `SAMPLER_INVALID`/`E2_INCONCLUSIVE`, không chạy study lớn.
5. Stage0:20 prompt × clean/key0 để kiểm tra. Các output này có phase/seed riêng,
   không nhập vào primary statistics dù prompt IDs thuộc pool đã freeze.
6. Stage1: group0, đủ8 keys0–7 và clean. Xuất `PILOT_ONLY`, rồi tự tiếp tục, không
   chọn key tốt, không điều chỉnh λ/threshold/Q theo kết quả pilot.
7. Stage2: bốn group còn lại, tổng **40000 teacher responses +5000 clean responses**.
   Mỗi response max512 new tokens, EOS nằm trong categorical và được giữ trong
   token audit. Response rỗng/ngắn/looping đều giữ nguyên. Stress23/59 thêm1000
   response mỗi key ở group0, báo cáo riêng. Không pool stress vào primary8.
8. Chấm primary bằng **returned text được re-tokenize**, replay clean frozen
   model một lần mỗi batch, tính V công khai rồi vector hóa correct+20 wrong keys.
   Native sampled-token control là phụ, không được thay primary.
9. Xuất báo cáo, từng key, utility/KL, examples đã chọn trước, ZIP review rồi dừng.
   Không tự chuyển LoRA, Experiment3, PSS, SaTML hoặc paraphrase API.

## Chạy dài, resume và chia shard

Mặc định ghi vào `experiment2_rs/run_012/`. Chạy lại cùng lệnh để resume. Thay
code/runtime/config/input sẽ bị từ chối; dùng `--output` mới để tạo phiên bản khác.
Tất cả dữ liệu cũ của E1 và frozen input trees được bảo vệ khỏi output lồng bên trong.

## Bản sửa lỗi `0 eligible`

V1 loại nhầm cả kho reserve chưa sử dụng và metadata eligible/removed như lịch sử
đã sử dụng. V2 phân biệt inventory với usage, vẫn loại cohort đã chọn, lịch sử dùng
trước đó và prompt thay thế đã tiêu thụ. Đồng thời đối chiếu source ID aliases và
hash chuẩn hóa lịch sử. Không giảm yêu cầu 5000 prompt, không đổi key, seed, λ hoặc
tiêu chí thí nghiệm. Input V1 và `experiment2_rs/run_001/` được giữ nguyên;
receipt V2 nằm riêng trong `e2_inputs_v2/`.

Đồng bộ cả code mới và `e2_inputs_v2/` lên server, rồi chạy lại file sh như trên.
Dataset đã tải có thể dùng lại từ cache Hugging Face. Bản sửa chỉ kiểm tra unit test
nhỏ trên dữ liệu giả; việc chọn đủ prompt và chạy model được thực hiện khi bạn chạy.

## Lịch sử KV parity trên A100 trước khi chuyển vLLM

**Bản hiện tại (run_005):** áp dụng FP64 accumulation bên trong nn.Linear cho cả
generation, native parity và verifier, giữ nguyên tensor trọng số/activation/KV
ở FP32. Không cache một bản trọng số FP64 thứ hai; chuyển tạm từng Linear mỗi lần
gọi để hạn chế VRAM, đổi lại có chi phí tốc độ. Readout/log-softmax/KL/scores vẫn
FP32. Ghi rõ policy trong resolved config, engine manifest và report; không gọi đây
là thuần FP32. Model/tokenizer, P/Q/L, key, seed, λ, cohort design và tolerance không
đổi. E1 code/artifacts và run 001–004 không thay đổi.

Lý do: trên worst batch run_004, FP64 Linear accumulation giảm max KV logprob error
từ 0.00028229 xuống 0.00004768; p99 decode1/2 lần lượt 0.00002098/0.00002289. Đây
là bằng chứng Linear roundoff đóng góp lớn, không phải chứng minh toàn bộ gate đạt.
Code mới vẫn kiểm tra đủ 10 prefill/20 decode và các gate còn lại trước study;
FAIL thì vẫn dừng, PASS mới tự tiếp tục. Chỉ kiểm tra CPU/mock nhỏ tại máy phát triển.

Đồng bộ `main_exp2.py` và toàn bộ code `klrd_rs/e2/` rồi chạy cùng
`bash run_experiment2.sh`. Không chạy lại run_004 hoặc sửa identity của run cũ.

Lịch sử chẩn đoán:

Run 002 đạt native rotation parity (max 0.00002098), nhưng KV/full-prefix parity
không đạt (max 0.00028229, p99 0.00007629). Đây là nhánh KV, không phải công thức
rotation. Chưa xác nhận được nguyên nhân số học trên cùng GPU; backend SDPA là
giả thuyết cần kiểm chứng. E2 nay dùng explicit eager attention cho cả prefill,
decode và verifier, vẫn FP32 và cùng trọng số. E1 loader không thay đổi.

Giữ nguyên ngưỡng max 0.0002/p99 0.00005 và toàn bộ key/seed/λ. Lần chạy mới ở
run_003; run_002 được bảo vệ khỏi ghi đè. Pool 5000 được chọn theo cùng input/seed,
không dùng kết quả sinh để chọn lại. Gate vẫn dừng nếu eager chưa đạt; lỗi và JSON
sẽ ghi riêng rotation/KV cùng prompt ID, token ID và decode step gây sai số lớn.
Chỉ kiểm tra mock/CPU nhỏ tại máy phát triển; parity Qwen/A100 cần bạn chạy lại.

Run 003 với eager vẫn lỗi KV (max 0.00028229, p99 0.00007439), vì vậy chưa xác
nhận SDPA là nguyên nhân. Run 004 giữ nguyên gate và tự chạy thêm một phép chẩn
đoán khi KV lỗi: đối chiếu lại đúng batch tệ nhất với tích lũy nn.Linear FP64 và
đầu ra mỗi Linear vẫn FP32. Không đổi trọng số, không phải full FP64 reference,
không dùng phép chẩn đoán này để cấp PASS hay sinh study. Xem thêm
`tests/kv_linear_precision_diagnostic.json`; pipeline vẫn dừng sau gate lỗi.
Run 003 được giữ nguyên và bảo vệ khỏi ghi đè. Đồng bộ `main_exp2.py` và
`klrd_rs/e2/gates.py`, rồi chạy cùng file sh để thu log chẩn đoán trên GPU.

Một job là một batch8 cố định trong `(owner, group, chunk)`. Categorical RNG riêng
cho từng owner/prompt/phase. Job JSON chứa tokens, EOS, seeds, response IDs và hash;
commit nguyên batch theo atomic rename. Sau interruption không gom lại các request
còn thiếu thành batch shape khác. Identity khởi tạo một lần có khóa; job và provenance
cũng có khóa khi chạy nhiều worker. Không chạy nhiều shard trên cùng GPU nếu không
đủ bộ nhớ; chỉ dùng cùng frozen runtime khi hợp nhất.

Các lệnh tùy chọn cho scheduler, không cần dùng khi chạy toàn bộ bằng file sh:

```bash
python main_exp2.py --pilot-only
python main_exp2.py --owner 2 --group 0
python main_exp2.py --owner clean --group 3 --chunk 7
python main_exp2.py --max-jobs 100
python main_exp2.py --analyze-only
```

`--max-jobs` giới hạn số job mới hoàn tất scoring trong lượt gọi; dữ liệu còn thiếu
được đánh dấu incomplete. Shard/analysis-only không có đủ toàn bộ cohort sẽ là
`E2_INCONCLUSIVE`, không giả là study đã xong. `--analyze-only` có thể chạy CPU trên
score caches đã tạo ở server, vẫn kiểm tra frozen design/group/prompt hashes.

## Báo cáo và private artifacts

Các thư mục theo đặc tả: provenance/, tests/, private/, scores/, utility/, report/.

- `scores/per_key_summary.csv`:16 hàng owner×Q, counts, per-key intervals,
  owner/clean/wrong distributions, identification, free-prefix KL và utility.
- `batch_owner_scores.csv`, `batch_clean_scores.csv`, `batch_wrong_key_scores.csv`:
  toàn bộ nhóm/p-values/cosines, zero-vector và trạng thái incomplete.
- `per_wrong_key_rates.csv`: tất cả20 wrong-key raw rates cho từng owner/Q.
- `closed_set_identification.csv`, `text_only_vs_native_control.csv` và stress controls.
- `utility/free_context_KL_by_key.csv`: sampled full-vocabulary KL(q||p0) trên prefix
  teacher thực sự đi qua, mean/p95/p99/max, tỷ lệ>0.05, so với E1 từng key và calibration.
  Hash mask1/16 theo prompt/position độc lập key. Key2 luôn được giữ lại.
- Utility: length/EOS/truncation/repetition/roundtrip, quality pairs100 prompt IDs
  blinded A/B, ít nhất10 preselected examples; answer mapping ở private/.
- Runtime: real-Qwen parity numbers, package versions, deterministic flags, peak
  VRAM, throughput, generation+verification seconds/GPU-hours và session wall time.
  Per-response timing không bao gồm discarded work của batch chưa commit; không
  xem đó là tổng chi phí hoàn chỉnh nếu có interruption.
- `KLRD_RS_experiment2_review.zip`: source/tests, configs, hash manifests, tables,
  selected redacted examples và report; **không** có key vectors, full raw outputs,
  private prompt pool, score caches, tokens/API credentials hay HPC login details.

Thiết kế chính có thể sinh tối đa23.04 triệu token (45000×512), chưa tính smoke và
stress. Đây là lượt chạy GPU dài; chi phí/throughput thực tế được ghi, không suy từ
CPU tests. λ/key bank không được đổi để tiết kiệm hoặc cải thiện TPR.

## Operational decision rules đã định trước

Promising chỉ khi >=32/40 Q1000 batches detect ở1e-4, >=6/8 keys đạt>=4/5, và không
có concern: mean wrong-key activation của bất kỳ owner >5% ở một trong hai alpha;
clean conservative detections>=2/5 cho một key; degeneration rate tăng>10 điểm%
so với clean; >=2 primary keys có p99 KL>0.05 hoặc bất kỳ key có>5% visited-prefix
KL samples vượt0.05; hay provenance không đầy đủ. Các ngưỡng này là cách vận hành
các tiêu chí định tính trong spec, đã chốt trước dữ liệu E2; không chỉnh theo pilot.
Chúng không là kiểm định fixed-key FPR hay chứng nhận utility.

Full study có <8/40 conservative và <8/40 nominal detections là E2_NO_GO; có tín hiệu
nhưng chưa đạt/ còn concern là E2_PARTIAL. Incomplete/invalid luôn E2_INCONCLUSIVE.
Mọi nhãn đều dừng để review; E2_PROMISING_FOR_LORA_REVIEW không cho phép tự huấn luyện.

Q1000 chỉ5 disjoint trials mỗi key; rates tăng theo20%. Các budgets nested, các keys
dùng chung prompt/clean bank, wrong keys dùng lại teacher text. Không bootstrap thành
trial độc lập, không pooled CI giả, không khẳng định rare fixed-key FPR đã được chứng minh.

## Kiểm thử và giới hạn chạy tại đây

```bash
python -m pytest -q
```

Suite có tiny real Qwen2 CPU tests cho cache/padding/native readout/EOS/RNG/resume,
hash bank E1, grouping5000 prompts, statistical counts và protected paths. Đây không
phải parity Qwen3B/CUDA hay kết quả E2. Máy nối với chat hiện CPU-only và không có
Qwen3B trong cache; **full GPU parity và45000 primary responses còn chờ bạn chạy trên
server bằng file sh**. Không tạo scientific GO hay số liệu Qwen từ toy tests.
