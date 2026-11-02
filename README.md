# KLRD-RS Experiment 1

Nguồn tái sử dụng đúng theo đính chính của bạn là thư mục **KLRD**:
`G:/Other computers/My Computer/Google Drive Offline/PHD/LLM_fingerprint/KLRD`.
Code, calibration, PCA và các cặp response/paraphrase hiện đã được lấy từ KLRD.
Runtime đọc bundle `inputs/` trong dự án, không cần đường dẫn máy này trên server.

Chạy trong thư mục dự án bằng môi trường Python đã cài dependencies:

```bash
bash run_experiment1.sh
```

Launcher hiện chạy numerical diagnostic và continuation theo yêu cầu review team:
`python main.py --continue-experiment1`. Xem `README_CONTINUATION.md` để biết các
artifact cũ cần giữ và toàn bộ tiêu chí FP64. Kết quả mới nằm ở
`experiment1_rs_continuation/`; `experiment1_rs/` cũ được bảo toàn.

Không đọc config ngoài, không kích hoạt/cài env,
không có đường dẫn Python cố định. Mang **toàn bộ thư mục dự án, gồm `inputs/`** lên
server. Các đường dẫn runtime được tính từ vị trí `main.py`.

Cần CUDA GPU và đủ VRAM cho Qwen2.5-3B FP32, activation replay và readout (khuyến nghị
24 GB trở lên; chưa đo peak VRAM trên máy này). Model/tokenizer tải từ Hugging Face
ở revision đã đóng băng trong đặc tả; server cần cache hoặc quyền truy cập model.
Dependencies được liệt kê trong `requirements.txt`; bạn tự cài môi trường.
Phiên bản package/model, input hash và các tham số phải được giữ khi resume.

Mặc định chạy duy nhất Experiment 1 A–H và legacy diagnostic tùy artifact.
Không huấn luyện, không gọi paraphrase API, không sinh câu mới từ teacher.
Toàn bộ token score dùng prompt nhìn thấy và response text được token hóa lại;
loại token special, EOS và pad. Calibration tiếp tục dùng cohort `train_gen` gốc.
Hai cohort diagnostic là đúng 2.500 clean-original/SCW pairs của historical
`followup_raw_llr_v1`, khớp cả prompt IDs và source response text; không lấy nhầm
clean_final ở run cha có cùng tên file nhưng khác nội dung.

Manifest calibration vẫn giữ đủ 2.000 response. Trong số đó 57 response có cờ
`text_roundtrip_equal=false`; pipeline không dùng prefix của chúng để fit Fisher,
chọn lambda hoặc đo local capacity. Từng partition ghi rõ IDs bị loại và số lượng
eligible, không thay cohort. Diagnostic giữ đủ 2.500 văn bản; 90 response có
token roundtrip lịch sử khác, nên verifier dùng canonical text-only reencoding
và không suy đoán token IDs nội bộ đã mất.

Kết quả ở `experiment1_rs/`: provenance, geometry, keys (IDs/hash), null, teacher,
legacy và report. Cache từng response/context/key nằm trong `private/cache` để
tiếp tục khi bị gián đoạn. Nếu đầu vào/code/cấu hình thay đổi, chương trình từ chối
resume; có thể chọn thư mục mới bằng `python main.py --output <new-directory>`.
Lambda được chọn một lần trên pooled calibration bank; không đổi theo validation key.

Các lệnh kiểm tra tùy chọn, không cần dùng khi chạy bình thường:

```bash
python main.py --audit-only --output input_audit
python main.py --smoke
python -m pytest -q
```

`--smoke` dùng model nhỏ CPU và ghi `SYNTHETIC_CHECKS_PASSED`; không chứng minh kết
quả Qwen. Audit chỉ kiểm tra dữ liệu. Các test không tải model hay gọi API.
Chạy Qwen thật và scientific gates trên server mới xác định được kết quả nghiên cứu.

Input có dữ liệu lịch sử và một legacy private key để diagnostic; `inputs/`, output
và cache đều được git-ignore. Không đưa các thư mục này vào repo công khai. Seed
bank nghiên cứu được công khai theo đặc tả; chúng không phải deployment secret.
Trên POSIX, dùng quyền truy cập riêng cho toàn bộ dữ liệu/output của thí nghiệm.

Đặc tả chuẩn: `KLRD_RS_FINAL_METHOD_AND_EXPERIMENT_1_SPEC.md`.
Spherical null là phát biểu ngẫu nhiên hóa trên key, không phải bảo đảm FPR của mọi
fixed key. Pilot dùng nhóm response không chồng lặp; Q=1000 chỉ là minh họa rất ít mẫu.
Local capacity là kỳ vọng oracle trên prefix sạch, chưa chứng minh TPR khi sinh câu,
LoRA implantation, fixed-key certification hay chống paraphrase/SaTML.
