# Phân tích Pipeline Cadebot AI

## 1. Tổng quan Pipeline

```
[Người dùng nói]
       ↓
[Ghi âm - Android MediaRecorder]
       ↓
[STT - PhoWhisper-large trên Laptop]
       ↓
[Text câu hỏi]
       ↓
[LLM - Qwen2.5-3B + LoRA trên Laptop]
       ↓
[Text câu trả lời]
       ↓
[Hiển thị chat + TTS đọc - Android TextToSpeech]
```

---

## 2. Chi tiết từng thành phần

### 2.1 Speech-to-Text (STT)

| Thông số | Giá trị |
|----------|---------|
| Model | `vinai/PhoWhisper-large` |
| Tổ chức | VinAI Research |
| Kiến trúc | Whisper-large-v2 fine-tuned tiếng Việt |
| Tham số | ~1.54B parameters |
| Kích thước | ~3.1 GB (FP32) / ~1.6 GB (FP16) |
| Ngôn ngữ | Tiếng Việt (tối ưu) |
| Phần cứng hiện tại | CPU (không có GPU) |
| Cách gọi | HTTP POST `/stt` — Android gửi file `.m4a` lên laptop |
| Thư viện | `transformers` pipeline + `soundfile` + `ffmpeg` |

**Lý do chọn PhoWhisper-large:** Độ chính xác tiếng Việt cao nhất trong họ PhoWhisper, được fine-tune đặc biệt cho giọng nói tiếng Việt đa vùng miền.

---

### 2.2 Language Model (Chat AI)

| Thông số | Giá trị |
|----------|---------|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Tổ chức | Alibaba Cloud |
| Kiến trúc | Transformer decoder, 3B parameters |
| Kích thước base | ~6 GB (FP16) |
| Phương pháp fine-tune | LoRA (Low-Rank Adaptation) |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Kích thước LoRA adapter | ~450 MB |

**Dữ liệu fine-tuning:**

| | Số lượng |
|--|--|
| Train set | 144 mẫu |
| Validation set | 26 mẫu |
| **Tổng** | **170 mẫu** |
| Số epoch | 5 |
| Loss ban đầu | 2.53 |
| Loss cuối | 0.16 |

**Nội dung dữ liệu:** Q&A về menu, giá, thành phần, gợi ý món, combo, gọi nhân viên — đặc thù cho Viva Reserve Coffee.

**Output format:** JSON có cấu trúc:
```json
{
  "intent": "MENU_QA | RECOMMENDATION | CALL_STAFF | ...",
  "confidence": 0.95,
  "answerText": "Câu trả lời đầy đủ hiển thị trong chat",
  "spokenText": "Câu ngắn gọn hơn",
  "recommendedItems": [{"menuItemId": "VR_LATTE", "reason": "..."}],
  "requiresHumanSupport": false
}
```

---

### 2.3 Text-to-Speech (TTS)

| Thông số | Giá trị |
|----------|---------|
| Engine | Android `TextToSpeech` API (built-in) |
| Ngôn ngữ | `vi_VN` (tiếng Việt) |
| Backend | Google TTS Engine (có sẵn trên Android) |
| Phần cứng | Chạy trực tiếp trên điện thoại |
| Độ trễ | < 0.5 giây |
| Nội dung đọc | `answerText` — khớp với text hiển thị trong chat |

---

### 2.4 Ghi âm

| Thông số | Giá trị |
|----------|---------|
| API | Android `MediaRecorder` |
| Format | MPEG-4 / AAC |
| Sample rate | 16,000 Hz |
| Channels | Mono |
| Bitrate | 64 kbps |
| Kích hoạt | Nhấn mic để bắt đầu, nhấn lại để dừng |

---

## 3. Phân tích tốc độ hiện tại (CPU)

**Môi trường:** Intel/AMD CPU, không có GPU, RAM 23GB

| Bước | Thời gian thực tế |
|------|------------------|
| Ghi âm (người dùng nói) | Tuỳ độ dài (~3–10 giây) |
| Upload audio lên server | < 1 giây |
| STT - PhoWhisper-large (CPU) | **~30 giây/giây audio** → 5s audio ≈ 150 giây |
| LLM - Qwen2.5-3B (CPU) | **~78 giây/response** (đo thực tế) |
| TTS đọc câu trả lời | < 0.5 giây |
| **Tổng (5 giây audio)** | **~230 giây (~4 phút)** |

**Bottleneck chính:** Cả STT lẫn LLM đều chạy trên CPU — không có hardware acceleration.

---

## 4. Phân tích tốc độ trên GPU — NVIDIA Jetson Orin Nano 8GB

### Thông số Jetson Orin Nano 8GB

| Thông số | Giá trị |
|----------|---------|
| CPU | 6-core Arm Cortex-A78AE @ 1.5 GHz |
| GPU | 1024-core NVIDIA Ampere GPU |
| AI Accelerator | 2x NVDLA (Deep Learning Accelerator) |
| Bộ nhớ | 8 GB LPDDR5 (unified CPU+GPU) |
| AI Performance | 40 TOPS (INT8) |
| Bandwidth | 68 GB/s |
| TDP | 5–10W |

---

### 4.1 STT - PhoWhisper-large trên Jetson Orin Nano

| | CPU hiện tại | Jetson Orin Nano (GPU) |
|--|--|--|
| Tốc độ | ~30x real-time chậm hơn | ~1–2x real-time |
| 5 giây audio | ~150 giây | **~5–10 giây** |
| Speedup | 1x (baseline) | **~15–30x nhanh hơn** |

> Whisper-large trên GPU Ampere chạy gần real-time. Với 1024 CUDA cores của Jetson Orin Nano, ước tính 5 giây audio xử lý trong 5–10 giây.

---

### 4.2 LLM - Qwen2.5-3B + LoRA trên Jetson Orin Nano

| | CPU hiện tại | Jetson Orin Nano (GPU FP16) | Jetson Orin Nano (GPU INT4) |
|--|--|--|--|
| Thời gian/response | ~78 giây | **~8–15 giây** | **~3–6 giây** |
| Tokens/giây | ~2–3 tok/s | ~15–25 tok/s | ~40–60 tok/s |
| Speedup | 1x | **~6–10x** | **~15–25x** |
| VRAM cần | 6 GB (hệ thống) | 6 GB | ~2 GB |

---

### 4.3 Vấn đề bộ nhớ — Jetson Orin Nano 8GB

Chạy đồng thời cả 2 model:

| Model | FP16 | INT4 (quantized) |
|-------|------|-----------------|
| PhoWhisper-large | ~1.6 GB | ~0.5 GB |
| Qwen2.5-3B + LoRA | ~6.0 GB | ~2.0 GB |
| **Tổng** | **~7.6 GB** ⚠️ gần đầy | **~2.5 GB** ✅ thoải mái |

**Kết luận:** Với FP16 có thể vừa đủ nhưng rủi ro OOM. **Nên dùng INT4 quantization (AWQ/GPTQ)** cho Qwen2.5-3B để đảm bảo ổn định và tăng tốc thêm.

---

### 4.4 Pipeline hoàn chỉnh trên Jetson Orin Nano

| Bước | CPU hiện tại | Jetson Orin Nano (INT4) |
|------|-------------|------------------------|
| Ghi âm | ~5 giây | ~5 giây |
| STT (5 giây audio) | ~150 giây | **~5–10 giây** |
| LLM response | ~78 giây | **~3–6 giây** |
| TTS | < 0.5 giây | < 0.5 giây |
| **Tổng** | **~235 giây** | **~15–20 giây** |
| **Speedup tổng** | 1x | **~12–15x nhanh hơn** |

---

## 5. Toàn bộ mẫu câu để test (144 mẫu từ tập train)

> Nói những câu dưới đây vào mic để test Cadebot. Được nhóm theo loại intent.

### 🍵 Hỏi về menu / thành phần / giá — MENU_QA (55 mẫu)

| # | Câu hỏi mẫu |
|---|-------------|
| 1 | Ice Blended Khoai Môn có gì trong đó? |
| 2 | Bánh croissant ăn với gì ngon? |
| 3 | Có dùng app đặt trước được không? |
| 4 | Cold Brew có thêm kem không? |
| 5 | Espresso có uống lạnh được không? |
| 6 | Trà đào giá bao nhiêu? |
| 7 | Cold Brew có nhiều caffeine không? |
| 8 | Có món ăn không? |
| 9 | Latte có caffeine không? |
| 10 | Khoai môn có thêm trân châu không? |
| 11 | Có thanh toán MoMo không? |
| 12 | Trà sữa Thái làm bằng sữa gì vậy? |
| 13 | Latte giá bao nhiêu vậy? |
| 14 | Trà sữa Jasmine làm từ gì vậy? |
| 15 | Dùng voucher như thế nào? |
| 16 | Bánh phô mai có lactose không? |
| 17 | Cold Brew pha như thế nào? |
| 18 | Foam kem là gì vậy? |
| 19 | Quán có gì ăn không? |
| 20 | Ice Blended Matcha bao nhiêu tiền vậy? |
| 21 | Signature của Viva là gì? |
| 22 | Uống ban đêm thì nên chọn gì, tôi sợ mất ngủ? |
| 23 | Americano có thêm sữa được không? |
| 24 | Americano có sữa không? |
| 25 | Cappuccino bọt nhiều không? |
| 26 | Americano có caffeine không? |
| 27 | Ice Blended có chỉnh ngọt được không? |
| 28 | Jasmine milk tea có thể bỏ sữa không? |
| 29 | Latte size L to không? |
| 30 | Trà sữa thái có gì đặc biệt? |
| 31 | Bánh croissant bao nhiêu vậy? |
| 32 | Gọi thêm món được không? |
| 33 | Hỏi chút, trà đào này uống lạnh thôi à? |
| 34 | Món nào bán chạy nhất? |
| 35 | Có thể giảm đường không? |
| 36 | Trà sữa Thái bao nhiêu? |
| 37 | Tea Sữa Jasmine có uống nóng không? |
| 38 | Latte uống nóng hay lạnh ngon hơn? |
| 39 | Cold brew có uống nóng không? |
| 40 | Cappuccino khác Latte chỗ nào? |
| 41 | Trà đào có thêm lô hội không? |
| 42 | Cadebot ơi, tôi muốn ăn gì đó |
| 43 | Matcha có caffeine không vậy? |
| 44 | Thêm trân châu được không? |
| 45 | Combo giá bao nhiêu? |
| 46 | Trà sữa jasmine ngọt không? |
| 47 | Thanh toán bằng gì vậy? |
| 48 | Espresso khác americano chỗ nào? |
| 49 | Trà đào có cà phê không? |
| 50 | Tôi không uống được cà phê, có gì không? |
| 51 | Trà đào có uống nóng được không? |
| 52 | Đặt xong bao lâu thì có món? |
| 53 | Đặt nhầm món có sửa được không? |
| 54 | Cái Cold Brew bao nhiêu tiền? |
| 55 | Vậy có gì không caffeine và uống lạnh không? |

---

### 🛒 Đặt món / thêm vào giỏ — ADD_TO_CART_DRAFT (27 mẫu)

| # | Câu mẫu |
|---|---------|
| 1 | 1 dâu tây đá xay không đường size M |
| 2 | Cho tôi 1 latte và 1 trà đào |
| 3 | Lấy tôi 1 americano đá, không đường |
| 4 | Cho tôi 1 cappuccino nóng |
| 5 | Đặt 1 jasmine lạnh thêm thạch dừa |
| 6 | Đặt 1 cold brew thêm kem, size L |
| 7 | 2 latte size L không đường ít đá |
| 8 | Cho tôi 3 trà đào size M |
| 9 | Cho tôi 1 trà sữa thái size L nhiều đá |
| 10 | Thêm 1 latte nữa vào giỏ giúp tôi |
| 11 | Tôi muốn 1 matcha đá xay size L |
| 12 | Tôi muốn 1 cheesecake và 1 trà jasmine nóng |
| 13 | Cho tôi 1 americano đá size L 0 đường |
| 14 | Tôi muốn 1 espresso |
| 15 | Cho tôi cái bán chạy nhất size L |
| 16 | Cho 1 trà sữa jasmine nóng ít đường |
| 17 | Cho tôi một latte size M ít đá 50% đường |
| 18 | 1 trà sữa thái 70% đường nhiều đá |
| 19 | Cho tôi 1 khoai môn và 1 croissant |
| 20 | Lấy tôi 2 trà đào lạnh |
| 21 | Tôi muốn 1 matcha thêm foam kem |
| 22 | 1 latte sữa ít ngọt không đá |
| 23 | Cho tôi combo cà phê và bánh |
| 24 | Tôi muốn đặt double espresso |
| 25 | Cho tôi một latte |
| 26 | Cho tôi 1 cold brew không đường |
| 27 | Cho tôi combo đôi |

---

### 💡 Xin gợi ý — RECOMMENDATION (23 mẫu)

| # | Câu mẫu |
|---|---------|
| 1 | Tôi đang hẹn hò, nên gọi gì? |
| 2 | Hôm nay tôi muốn thử gì mới lạ? |
| 3 | Trời nóng uống gì mát mẻ nhỉ? |
| 4 | Muốn uống gì trái cây mát lạnh |
| 5 | Uống kèm bánh thì chọn gì? |
| 6 | Uống gì không quá ngọt, có caffeine? |
| 7 | Tôi không biết chọn gì, Cadebot giúp tôi với |
| 8 | Buổi sáng uống gì tỉnh táo? |
| 9 | Cadebot thích uống gì nhất? |
| 10 | Tôi muốn uống gì không caffeine, lạnh, ít ngọt |
| 11 | Tôi bé uống trà sữa Thái ngọt quá, có gì ngọt vừa không? |
| 12 | Gợi ý gì đó cho người lần đầu đến Viva |
| 13 | Chiều chiều uống gì thư giãn? |
| 14 | Uống gì rẻ mà ngon? |
| 15 | Uống gì cho bữa sáng kèm bánh? |
| 16 | Gợi ý cho tôi món trà không cà phê nào ngon? |
| 17 | Tôi đang mệt, uống gì cho tỉnh? |
| 18 | Tôi muốn uống gì có sữa không caffeine |
| 19 | Tôi muốn uống gì cần tập trung làm việc? |
| 20 | Gợi ý cho tôi món gì ngon nhất đi |
| 21 | Tôi muốn gợi ý món không ngọt |
| 22 | Muốn uống gì màu đẹp chụp hình |
| 23 | Cho 2 người uống thì gọi gì? |

---

### 🔔 Gọi nhân viên — CALL_STAFF (12 mẫu)

| # | Câu mẫu |
|---|---------|
| 1 | Cho tôi gặp nhân viên |
| 2 | Cho tôi hỏi về thông tin thành viên |
| 3 | Tôi cần hỗ trợ |
| 4 | Robot ơi tôi cần giúp đỡ |
| 5 | Cần hỗ trợ gấp |
| 6 | Cần người phục vụ |
| 7 | Tôi muốn khiếu nại về đơn hàng |
| 8 | Hỏi chị nhân viên giúp tôi với |
| 9 | Gọi nhân viên giúp tôi |
| 10 | Gọi giúp tôi anh nhân viên |
| 11 | Ơi robot ơi, gọi nhân viên đi |
| 12 | Tôi muốn nói chuyện với người thật |

---

### 🎁 Khuyến mãi / combo — PROMOTION_QA (9 mẫu)

| # | Câu mẫu |
|---|---------|
| 1 | Happy hour mấy giờ vậy? |
| 2 | Combo cà phê và bánh gồm những gì? |
| 3 | Dùng voucher có giảm giá thêm không? |
| 4 | Combo đôi có áp dụng với ice blended không? |
| 5 | Có chương trình khuyến mãi nào đang diễn ra không? |
| 6 | Combo đôi tính như thế nào vậy? |
| 7 | Có combo không? |
| 8 | Hôm nay có ưu đãi gì không? |
| 9 | Mua 2 tặng 1 có không? |

---

### ❓ Ngoài phạm vi — FALLBACK (18 mẫu)

> Cadebot sẽ trả lời chưa có thông tin và đề nghị hỏi nhân viên.

| # | Câu mẫu |
|---|---------|
| 1 | Cho tôi mượn sạc điện thoại |
| 2 | Cadebot được làm bởi ai vậy? |
| 3 | Uống latte có béo không? |
| 4 | Quán đóng cửa mấy giờ? |
| 5 | Viva lấy cà phê từ đâu vậy? |
| 6 | Viva có ship đồ uống không? |
| 7 | Giảm giá cho tôi được không? |
| 8 | Viva Coffee có mấy chi nhánh? |
| 9 | Cho tôi số điện thoại quán |
| 10 | Tôi muốn mua đồ uống về nhà, có ship không? |
| 11 | Kể chuyện cười đi |
| 12 | Tôi muốn mua nguyên liệu về nhà tự pha |
| 13 | Cà phê bao nhiêu calo? |
| 14 | Máy POS bị lỗi |
| 15 | Cadebot có học không? |
| 16 | Bài nhạc đang phát là gì vậy? |
| 17 | Thời tiết hôm nay thế nào? |
| 18 | Tôi bị dị ứng gluten, có ăn bánh được không? |

---

## 6. Giải pháp Cloud/API để đạt ≤ 3 giây (không cần tự mua GPU)

> ⚠️ **KHẢO SÁT PHƯƠNG ÁN — KHÔNG PHẢI KIẾN TRÚC ĐANG CHẠY.**
> Mục này so sánh các hướng nâng cấp *đã cân nhắc và chưa chọn*. Hệ thống thật
> tự host toàn bộ: STT bằng PhoWhisper-large và LLM bằng Qwen2.5-3B + LoRA, đều
> chạy trong `cadebot-api`, không gọi dịch vụ AI bên thứ ba nào. Mọi cái tên nhà
> cung cấp dưới đây (Groq, RunPod, GPT-4o-mini, Gemini…) chỉ là đối tượng so
> sánh trên giấy. Kiến trúc thật xem [architecture.md](architecture.md).

> Cả 3 hướng đều rất rẻ vì dùng **serverless** — chỉ tính tiền khi có request, không tốn tiền lúc quán đóng cửa.

### Hướng 1 — Groq STT + Serverless GPU cho LLM

Dùng Groq cho STT (nhanh, rẻ) + thuê GPU serverless chỉ để chạy model fine-tuned Qwen2.5-3B + LoRA.

| Thành phần | Service | Tốc độ | Giá |
|---|---|---|---|
| STT | Groq `whisper-large-v3-turbo` | ~0.3–0.5s | $0.04/giờ audio |
| LLM | Modal.com hoặc RunPod (A10G) | ~0.5–1s | ~$0.69–1.10/giờ compute |
| **Tổng** | | **~1–1.5s ✅** | |

**Chi phí thực tế (100 khách/ngày):**
- STT: 100 × 10s audio × 30 ngày = ~8.3 giờ audio/tháng → **~$0.33/tháng**
- LLM: 100 request × ~1s compute × 30 ngày = ~0.83 giờ GPU/tháng → **~$0.91/tháng**
- **Tổng: ~$1.5–2/tháng** 🎉

**Ưu điểm:** Giữ được model fine-tuned + PhoWhisper tốt nhất cho tiếng Việt.  
**Nhược điểm:** STT dùng Groq whisper (không phải PhoWhisper-large), có thể kém chính xác hơn chút với giọng địa phương.

---

### Hướng 2 — Tự deploy cả 2 model trên Serverless GPU

Deploy PhoWhisper-large + Qwen2.5-3B + LoRA lên cùng 1 GPU serverless — giữ nguyên toàn bộ pipeline hiện tại, chỉ đổi nơi chạy.

| Service | GPU | Tốc độ tổng | Giá/giờ compute | Chi phí/tháng |
|---|---|---|---|---|
| **RunPod Serverless** | RTX 4090 24GB | ~0.5–1s ✅ | $0.69 | **~$1.5–2** |
| **Modal.com** | A10G 24GB | ~0.8–1.5s ✅ | $1.10 | **~$2–3** |
| **Replicate** | A40 48GB | ~0.5–1s ✅ | ~$1.40 | **~$3–4** |
| AWS EC2 g5.xlarge | A10G 24GB | ~0.8–1.5s ✅ | $1.006 | **~$2.5–3** |

**Ưu điểm:** Giữ 100% PhoWhisper-large + model fine-tuned, chính xác nhất.  
**Nhược điểm:** Phải tự setup deploy (phức tạp hơn hướng 1 và 3).

---

### Hướng 3 — Dùng API thương mại hoàn toàn

Không tự host model nào. Dùng Groq cho STT + LLM thương mại với system prompt chứa thông tin menu Viva Coffee.

| STT | LLM | Tốc độ | Chi phí/tháng (100 khách/ngày) |
|---|---|---|---|
| Groq `whisper-large-v3-turbo` | **GPT-4o-mini** | ~1–2s ✅ | **~$0.60** |
| Groq `whisper-large-v3-turbo` | **Gemini 2.0 Flash** | ~0.8–1.5s ✅ | **~$0.30** |
| Groq `whisper-large-v3-turbo` | **Groq Llama-3.3-70B** | ~0.5–1s ✅ | **~$0.40** |

**Chi phí chi tiết (100 khách/ngày, ~200 tokens input + ~100 tokens output):**
- STT (Groq turbo): ~$0.33/tháng
- LLM GPT-4o-mini: 100×30×300 tokens = 900K tokens → ~$0.27/tháng
- **Tổng: ~$0.60/tháng** 🎉

**Ưu điểm:** Rẻ nhất, đơn giản nhất, không cần maintain gì.  
**Nhược điểm:** Mất model fine-tuned → có thể kém chính xác với từ ngữ đặc thù Viva Coffee. Bù đắp bằng system prompt tốt.

---

### Pipeline của từng hướng

**Hướng 1 — Groq STT + Serverless GPU LLM:**
```
[Người dùng nói]
      ↓
[Ghi âm - Android MediaRecorder]
      ↓ upload file .m4a
[Groq Cloud - whisper-large-v3-turbo]  ~0.3–0.5s
      ↓ text
[RunPod/Modal - Qwen2.5-3B + LoRA]     ~0.5–1s
      ↓ answerText
[Android TextToSpeech]                  ~0.2s
Tổng: ~1–1.5s
```

**Hướng 2 — Full Serverless GPU:**
```
[Người dùng nói]
      ↓
[Ghi âm - Android MediaRecorder]
      ↓ upload file .m4a
[RunPod/Modal GPU]
  ├─ PhoWhisper-large (STT)            ~0.5–1s
  ↓ text
  └─ Qwen2.5-3B + LoRA (LLM)          ~0.5–1s
      ↓ answerText
[Android TextToSpeech]                  ~0.2s
Tổng: ~0.5–1.5s
```

**Hướng 3 — Full API thương mại:**
```
[Người dùng nói]
      ↓
[Ghi âm - Android MediaRecorder]
      ↓ upload file .m4a
[Groq Cloud - whisper-large-v3-turbo]  ~0.3–0.5s
      ↓ text
[Gemini Flash / GPT-4o-mini]           ~0.5–1.5s
      ↓ answerText
[Android TextToSpeech]                  ~0.2s
Tổng: ~0.8–2s
```

---

### Chi phí theo quy mô — 100 vs 500 khách/ngày

> Giả định: 10 giây audio/khách, ~200 input + ~100 output tokens/request, 30 ngày/tháng.

| | 100 khách/ngày | 500 khách/ngày |
|---|---|---|
| Audio/tháng | ~8.3 giờ | ~41.7 giờ |
| LLM requests/tháng | 3,000 | 15,000 |
| Input tokens/tháng | 0.6M | 3.0M |
| Output tokens/tháng | 0.3M | 1.5M |

**Hướng 1 — Groq STT + RunPod LLM (A10G):**

| Chi phí | 100 khách/ngày | 500 khách/ngày |
|---|---|---|
| Groq whisper-turbo | ~$0.33 | ~$1.67 |
| RunPod RTX 4090 (LLM) | ~$0.91 | ~$2.88 |
| **Tổng** | **~$1.24/tháng** | **~$4.54/tháng** |

**Hướng 2 — Full Serverless GPU:**

| Service | 100 khách/ngày | 500 khách/ngày |
|---|---|---|
| RunPod RTX 4090 | ~$1.04 | ~$4.31 |
| Modal A10G | ~$1.65 | ~$6.88 |

**Hướng 3 — Full API thương mại:**

| LLM | 100 khách/ngày | 500 khách/ngày |
|---|---|---|
| Gemini 2.0 Flash | ~$0.47 | ~$2.34 |
| GPT-4o-mini | ~$0.60 | ~$3.02 |
| Groq Llama-3.3-70B | ~$0.92 | ~$4.62 |

> Groq STT (~$0.33 → ~$1.67) tính chung vào Hướng 3.

---

### So sánh tổng hợp 3 hướng

| | Hướng 1 | Hướng 2 | Hướng 3 |
|---|---|---|---|
| **Tốc độ** | ~1–1.5s ✅ | ~0.5–1.5s ✅ | ~0.8–2s ✅ |
| **Chi phí/tháng** | ~$2–3 | ~$2–4 | ~$0.30–0.60 |
| **Giữ fine-tuned model** | ✅ LLM | ✅ Cả 2 | ❌ |
| **Độ chính xác tiếng Việt STT** | ⚠️ Groq whisper | ✅ PhoWhisper | ⚠️ Groq whisper |
| **Độ phức tạp setup** | Trung bình | Cao | Thấp |
| **Phụ thuộc internet** | Có | Có | Có |

**Khuyến nghị:** Hướng 2 (RunPod/Modal) nếu ưu tiên chính xác; Hướng 3 (Gemini Flash) nếu ưu tiên đơn giản và rẻ nhất.

---

## 7. Lộ trình nâng cấp đề xuất

| Giai đoạn | Phần cứng | Thay đổi | Tốc độ dự kiến |
|-----------|-----------|----------|----------------|
| **Hiện tại** | CPU laptop | FP16, không GPU | ~4 phút/turn |
| **Giai đoạn 1** | Jetson Orin Nano 8GB | INT4 quantization | **~15–20 giây/turn** |
| **Giai đoạn 2** | Jetson Orin Nano 8GB | TensorRT optimization | **~8–12 giây/turn** |
| **Giai đoạn 3** | NVIDIA GPU rời (RTX 3060+) | FP16, full speed | **~2–4 giây/turn** |

> **Giai đoạn 1 → 2:** Dùng `tensorrt_llm` hoặc `torch.compile()` để tối ưu inference graph.  
> **Giai đoạn 3:** Với GPU rời từ RTX 3060 trở lên, toàn bộ pipeline chạy near real-time.
