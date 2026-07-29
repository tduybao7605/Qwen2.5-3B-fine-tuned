# Kế hoạch triển khai Cadebot — Knowledge Base + Dynamic RAG

## Bối cảnh
Cadebot cần trả lời câu hỏi khách quán cà phê dựa trên knowledge base (menu, giá, khuyến mãi). Nếu câu hỏi ngoài phạm vi kiến thức, bot phải báo không trả lời được thay vì bịa. Kiến trúc: Dify (self-hosted) lo phần build/quản lý knowledge base, dynamic RAG tự viết lo phần truy vấn + trả lời + logic out-of-scope.

---

## Giai đoạn 1 — Hạ tầng Dify (self-hosted)

- [ ] Cài Docker (nếu máy/server chưa có)
- [ ] Clone repo Dify, copy `.env.example` → `.env`
- [ ] Chọn vector DB: **Qdrant** (khuyến nghị, dễ query trực tiếp) hoặc pgvector
- [ ] Sửa `.env`: đặt `VECTOR_STORE=qdrant` + các biến `QDRANT_*` tương ứng
- [ ] `docker compose up -d`, kiểm tra Dify + Qdrant chạy được (Qdrant lắng nghe cổng 6333)

## Giai đoạn 2 — Chốt model

- [ ] **LLM trả lời**: Qwen2.5-7B-Instruct
- [ ] **Embedding model** (dùng để build KB — không phải Qwen2.5-7B):
  - Nhẹ/edge: `Qwen3-Embedding-0.6B`
  - Server, không giới hạn phần cứng: `Qwen3-Embedding-4B` hoặc `-8B`
  - Ghi lại rõ **tên model + số chiều vector (dimension)** — dùng xuyên suốt, không đổi giữa chừng
- [ ] **STT/TTS**: chọn model riêng, không liên quan tới embedding
- [ ] Trong Dify: **Settings → Model Providers** → thêm provider chứa embedding model đã chọn (qua API hoặc self-host Ollama/Xinference/HuggingFace)

## Giai đoạn 3 — Tạo Knowledge Base demo

- [ ] Soạn dữ liệu mẫu: menu, giá, mô tả món, khuyến mãi (dạng text/docx/excel)
- [ ] Trong Dify: **Knowledge → Create Knowledge** → chọn chế độ **High Quality** → chọn embedding model đã cấu hình
- [ ] Upload dữ liệu mẫu, kiểm tra chunk ra hợp lý không
- [ ] Dùng **Retrieval Testing** trong Dify: thử vài câu hỏi trong phạm vi và ngoài phạm vi, ghi lại điểm số (score) để chọn **Score Threshold** phù hợp

## Giai đoạn 4 — Database demo (menu + khuyến mãi)

- [ ] Dựng DB demo (SQLite/PostgreSQL) với 2 bảng: `menu_items`, `promotions`
- [ ] Viết hàm `get_menu_data()` tách riêng — đọc dữ liệu từ DB, trả về danh sách món/khuyến mãi
- [ ] Viết script convert dữ liệu → đoạn text mô tả (món, giá, khuyến mãi)
- [ ] Viết script gọi **Dify API** (create/update document) để đẩy dữ liệu vào Knowledge base
- [ ] Test: sửa 1 dòng trong DB demo → chạy script → kiểm tra KB trong Dify cập nhật đúng

## Giai đoạn 5 — Dynamic RAG + logic out-of-scope

- [ ] Quyết định cách RAG lấy dữ liệu: đọc thẳng Qdrant, hay gọi Dify Knowledge Retrieval API (nếu dùng Chatflow của Dify)
- [ ] Đảm bảo embedding model dùng để encode câu hỏi user **trùng khớp** với model đã dùng để embed KB
- [ ] Cài logic: nếu không có chunk nào vượt Score Threshold → trả lời cố định "ngoài phạm vi, không thể trả lời", không gọi LLM
- [ ] Nếu có chunk phù hợp → đưa context cho Qwen2.5-7B, prompt ràng buộc "chỉ trả lời dựa trên context, không tự bịa"
- [ ] Test bộ câu hỏi: trong phạm vi (menu, giá, khuyến mãi) và ngoài phạm vi (thời tiết, chính trị...) để kiểm tra bot từ chối đúng lúc

## Giai đoạn 6 — Tích hợp STT/TTS

- [ ] Kết nối STT: giọng nói khách → text → đưa vào pipeline RAG ở Giai đoạn 5
- [ ] Kết nối TTS: câu trả lời text → giọng nói phản hồi khách
- [ ] Test end-to-end: hỏi bằng giọng nói → nhận câu trả lời bằng giọng nói, đúng nội dung

## Giai đoạn 7 — Chuyển sang database thật (khi quán cung cấp)

- [ ] Xin thông tin kết nối DB thật của quán (loại DB, bảng menu, quyền truy cập hoặc API)
- [ ] Sửa lại **chỉ phần bên trong `get_menu_data()`** để trỏ đúng DB/API thật — giữ nguyên toàn bộ logic sync với Dify
- [ ] Test lại toàn bộ pipeline với dữ liệu thật
- [ ] Thiết lập cách đồng bộ khi giá/khuyến mãi đổi: nút bấm thủ công hoặc cron chạy định kỳ (vì tần suất đổi thấp, không cần real-time)

---

## Lưu ý xuyên suốt
- Đổi embedding model giữa chừng = phải embed lại toàn bộ KB → chốt model từ Giai đoạn 2, tránh đổi sau.
- Bảng đơn hàng/giao dịch của POS **không** đưa vào KB — chỉ đồng bộ bảng menu/giá/khuyến mãi.
- High Quality (có embedding) không hạ được về Economical (keyword-only) sau khi đã tạo KB.
