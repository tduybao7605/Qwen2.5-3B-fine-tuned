# Cadebot — Tài liệu Tích hợp STT & Model API

## Tổng quan những gì đã thêm

### 1. Groq Whisper STT (Speech-to-Text)
Thay thế Android built-in `SpeechRecognizer` bằng Groq Whisper API.  
Flow: **Ghi âm (MediaRecorder) → gửi file âm thanh → Groq trả text → điền vào input.**

### 2. Kết nối Model thật trên Laptop
Thay thế `MockAiService` bằng `CadebotApiService` gọi HTTP đến server FastAPI chạy model Qwen2.5-3B + LoRA.

### 3. FastAPI Server (serve_model.py)
Script chạy model trên laptop, expose endpoint `/chat` cho app Android gọi tới.

---

## Các file đã thay đổi / tạo mới

| File | Loại | Mô tả |
|------|------|--------|
| `Cadebot_UI/app/src/main/AndroidManifest.xml` | Sửa | Thêm `RECORD_AUDIO`, `INTERNET` permission |
| `Cadebot_UI/gradle/libs.versions.toml` | Sửa | Thêm OkHttp `4.12.0` |
| `Cadebot_UI/app/build.gradle.kts` | Sửa | Bật `buildConfig`, đọc `local.properties`, thêm OkHttp dep |
| `Cadebot_UI/local.properties` | Sửa | Thêm `groq.api.key` và `cadebot.api.url` |
| `data/remote/CadebotApiService.kt` | **Mới** | HTTP client gọi model server |
| `ui/ai/GroqSttService.kt` | **Mới** | Ghi âm + gọi Groq Whisper API |
| `di/AppModule.kt` | Sửa | Provide `CadebotApiService` thay `MockAiService` |
| `ui/ai/AiViewModel.kt` | Sửa | Dùng `CadebotApiService`, thêm state `isListening`/`isTranscribing` |
| `ui/ai/AiScreen.kt` | Sửa | Tích hợp `GroqSttService`, 3 trạng thái mic UI |
| `serve_model.py` | **Mới** | FastAPI server chạy model Qwen2.5-3B + LoRA |

---

## API 1 — Groq Whisper (Speech-to-Text)

### Endpoint
```
POST https://api.groq.com/openai/v1/audio/transcriptions
```

### Authentication
```
Authorization: Bearer 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
```

### Request (multipart/form-data)
| Field | Value | Ghi chú |
|-------|-------|---------|
| `file` | file âm thanh `.m4a` | Ghi bởi `MediaRecorder` (AAC, 16kHz, 64kbps) |
| `model` | `whisper-large-v3-turbo` | Model Whisper nhanh nhất của Groq |
| `language` | `vi` | Tiếng Việt |
| `response_format` | `json` | Trả JSON |

### Response
```json
{
  "text": "nội dung người dùng đã nói"
}
```

### Code tham chiếu
`GroqSttService.kt` → hàm `suspend fun transcribe(apiKey: String): String?`

```kotlin
// Ghi âm
sttService.startRecording()          // bắt đầu ghi

// Dừng và gửi Groq
sttService.stopRecording()
val text = sttService.transcribe(BuildConfig.GROQ_API_KEY)
```

---

## API 2 — Cadebot Model Server (trên Laptop)

### Setup server
```bash
pip install fastapi uvicorn transformers peft torch accelerate
python serve_model.py
```

Server chạy tại `http://0.0.0.0:8000`.

### Endpoints

#### `GET /health`
Kiểm tra server còn sống không.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

#### `POST /chat`
Gửi tin nhắn, nhận phản hồi từ model Qwen2.5-3B + LoRA.

**Request body:**
```json
{
  "message": "Latte có caffeine không?",
  "history": [
    { "role": "user", "content": "Xin chào" },
    { "role": "assistant", "content": "Xin chào bạn!" }
  ]
}
```

| Field | Type | Mô tả |
|-------|------|--------|
| `message` | `string` | Tin nhắn hiện tại của người dùng |
| `history` | `array` | Lịch sử hội thoại (tối đa 8 messages gần nhất được dùng) |

**Response:**
```json
{
  "response": "{\"intent\":\"MENU_QA\",\"confidence\":0.95,\"answerText\":\"Latte có caffeine vì được pha từ espresso...\",\"recommendedItems\":[{\"menuItemId\":\"VR_LATTE\",\"reason\":\"...\"}],...}"
}
```

> **Lưu ý:** `response` là chuỗi JSON do model sinh ra.  
> Android app parse lại để lấy `answerText` và `recommendedItems`.

### Cấu trúc JSON mà model trả về (bên trong `response`)
```json
{
  "intent": "MENU_QA | RECOMMENDATION | ADD_TO_CART_DRAFT | PROMOTION_QA | CALL_STAFF | FALLBACK",
  "confidence": 0.95,
  "answerText": "Câu trả lời hiển thị trong chat",
  "spokenText": "Câu trả lời ngắn hơn cho TTS",
  "recommendedItems": [
    { "menuItemId": "VR_LATTE", "reason": "Lý do gợi ý" }
  ],
  "draftCartItems": [],
  "requiresHumanSupport": false,
  "sourceIds": ["menu:VR_LATTE"]
}
```

### Code tham chiếu
`CadebotApiService.kt` → hàm `suspend fun processQuery(message, history): AiMessage`

---

## Cách cấu hình và Build APK

### Bước 1 — Lấy IP của laptop
```bash
# Linux/Mac
ip addr show | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig
```
Ví dụ IP: `192.168.1.105`

### Bước 2 — Cập nhật local.properties
Mở file `Cadebot_UI/local.properties`, sửa dòng:
```
cadebot.api.url=http://192.168.1.105:8000
```
Thay `192.168.1.105` bằng IP thật của laptop.

### Bước 3 — Chạy server trên laptop
```bash
cd /path/to/Qwen2.5-3B-fine-tuned
pip install fastapi uvicorn transformers peft torch accelerate
python serve_model.py
```
Đợi đến khi thấy: `✅ Model ready!`

### Bước 4 — Build APK bằng Android Studio
1. Mở Android Studio
2. File → Open → chọn thư mục `Cadebot_UI/`
3. Đợi Gradle sync
4. Menu: **Build → Build Bundle(s) / APK(s) → Build APK(s)**
5. APK xuất tại: `app/build/outputs/apk/debug/app-debug.apk`

> **Yêu cầu:** Điện thoại và laptop phải cùng mạng WiFi để app kết nối được server.

---

## Luồng hoạt động toàn bộ

```
[User nói vào mic]
       ↓
[Android MediaRecorder ghi âm → file .m4a]
       ↓
[GroqSttService gửi lên Groq Whisper API]
       ↓  POST /audio/transcriptions
[Groq trả về text tiếng Việt]
       ↓
[Text điền vào input field]
       ↓  (User nhấn Send)
[CadebotApiService POST /chat đến Laptop Server]
       ↓
[FastAPI → Qwen2.5-3B + LoRA inference]
       ↓
[Model trả JSON có answerText + recommendedItems]
       ↓
[Android parse JSON → hiển thị chat bubble + gợi ý món]
```

---

## Các thư viện / dependencies đã thêm

| Library | Version | Mục đích |
|---------|---------|----------|
| `com.squareup.okhttp3:okhttp` | `4.12.0` | HTTP client cho Groq API và model server |
| `android.permission.RECORD_AUDIO` | built-in | Quyền ghi âm |
| `android.permission.INTERNET` | built-in | Quyền gọi API |
| `android.media.MediaRecorder` | built-in Android | Ghi âm thành file .m4a |
| `org.json.JSONObject` | built-in Android | Parse JSON response từ model |

---

## Lưu ý bảo mật

- `local.properties` đã có trong `.gitignore` → API key **không bị commit lên git**
- API key được inject vào `BuildConfig` lúc build → không hardcode trong source
- Nên dùng biến môi trường hoặc Android Keystore cho production

---

*Tài liệu này được tạo tự động sau khi tích hợp Groq STT và Cadebot Model API.*
