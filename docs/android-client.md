# Cadebot — Android client

App Android (Jetpack Compose) cho tablet đặt tại bàn. Ghi âm, gửi lên Cadebot
API để nhận dạng và trả lời, rồi đọc câu trả lời bằng TTS của Android.

**Kiến trúc: chỉ một backend.** Cả speech-to-text lẫn hội thoại đều gọi về
Cadebot API tự host. App **không** dùng dịch vụ AI bên thứ ba nào và **không
cần API key nào**.

```
[User nói vào mic]
       ↓
[MediaRecorder ghi âm → file .m4a (AAC, 16kHz)]
       ↓  POST /stt   (multipart)
[Cadebot API — PhoWhisper-large]
       ↓
[Text tiếng Việt điền vào ô nhập]
       ↓  (User nhấn Send)   POST /chat
[Cadebot API — retrieval (Dify+Qdrant) → Qwen2.5-3B + LoRA]
       ↓
[JSON: intent, answerText, spokenText, recommendedItems, sourceIds]
       ↓
[Hiển thị chat bubble + gợi ý món, đọc spokenText bằng TextToSpeech]
```

Toàn bộ lưu lượng đi tới một địa chỉ duy nhất: `BuildConfig.CADEBOT_API_URL`.

---

## Các file chính

| File | Vai trò |
|------|---------|
| `ui/ai/SttService.kt` | Ghi âm bằng `MediaRecorder`, POST file lên `/stt`, trả text |
| `data/remote/CadebotApiService.kt` | HTTP client gọi `/chat`, parse JSON của model |
| `ui/ai/AiScreen.kt` | Màn hình hội thoại, 3 trạng thái mic, phát TTS |
| `ui/ai/AiViewModel.kt` | State `isListening` / `isTranscribing`, gọi `CadebotApiService` |
| `di/AppModule.kt` | Provide `CadebotApiService` thay cho `MockAiService` |
| `app/build.gradle.kts` | Bật `buildConfig`, đọc `cadebot.api.url` từ `local.properties` |
| `AndroidManifest.xml` | Quyền `RECORD_AUDIO`, `INTERNET` |

---

## Cấu hình

Chỉ có **một** giá trị cần đặt.

```bash
cd Cadebot_UI
cp local.properties.example local.properties
```

Sửa `cadebot.api.url` cho đúng nơi server đang chạy:

| Tình huống | Giá trị |
|---|---|
| Android emulator, server trên cùng máy | `http://10.0.2.2:8000` |
| Thiết bị thật, cùng mạng LAN | `http://192.168.1.x:8000` |
| Server đã expose qua tunnel / reverse proxy | `https://your-domain.example` |

Lấy IP LAN của máy chạy server:

```bash
ip addr show | grep "inet " | grep -v 127.0.0.1   # Linux/macOS
ipconfig                                           # Windows
```

`local.properties` nằm trong `.gitignore` — không commit file này.

---

## Build APK

```bash
cd Cadebot_UI
./gradlew assembleDebug
# APK: app/build/outputs/apk/debug/app-debug.apk
```

Hoặc bằng Android Studio: mở thư mục `Cadebot_UI/`, đợi Gradle sync, rồi
**Build → Build Bundle(s) / APK(s) → Build APK(s)**.

> Nếu dùng IP LAN thì điện thoại và máy chủ phải cùng mạng WiFi.

Server phải đang chạy trước khi mở app — xem [deployment.md](deployment.md).

---

## API mà app gọi

Hợp đồng đầy đủ ở [api-reference.md](api-reference.md); dưới đây là phần app
thực sự dùng.

### `POST /stt` — Speech-to-Text

Multipart, field `file` là bản ghi `.m4a`. Server dùng `ffmpeg` chuyển sang WAV
16 kHz mono rồi đưa qua PhoWhisper-large.

```json
{ "text": "nội dung người dùng đã nói" }
```

Code: `SttService.kt` → `suspend fun transcribe(serverUrl: String): String?`

```kotlin
sttService.startRecording()
// ...
sttService.stopRecording()
val text = sttService.transcribe(BuildConfig.CADEBOT_API_URL)
```

### `POST /chat` — Hội thoại

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
| `message` | `string` | Tin nhắn hiện tại |
| `history` | `array` | Lịch sử hội thoại (server chỉ dùng 8 lượt gần nhất) |
| `use_rag` | `bool` | Không bắt buộc. App không gửi → server mặc định **bật** |

**Response:**

```json
{
  "response": "{\"intent\":\"MENU_QA\",\"answerText\":\"...\",\"sourceIds\":[\"menu:VR_LATTE\"]}",
  "retrieval": { "in_scope": true, "top_score": 0.71, "threshold": 0.51, "sourceIds": ["menu:VR_LATTE"] }
}
```

> **Lưu ý:** `response` là **chuỗi** chứa JSON, không phải object lồng nhau.
> App phải `JSONObject(...)` lần nữa để lấy `answerText` / `recommendedItems`.

### Cấu trúc JSON bên trong `response`

```json
{
  "intent": "MENU_QA | RECOMMENDATION | ADD_TO_CART_DRAFT | PROMOTION_QA | CALL_STAFF | FALLBACK",
  "confidence": 0.95,
  "answerText": "Câu trả lời hiển thị trong chat",
  "spokenText": "Câu trả lời ngắn hơn cho TTS",
  "recommendedItems": [{ "menuItemId": "VR_LATTE", "reason": "Lý do gợi ý" }],
  "draftCartItems": [],
  "requiresHumanSupport": false,
  "sourceIds": ["menu:VR_LATTE"]
}
```

Câu ngoài phạm vi kiến thức trả `intent: "FALLBACK"` với `sourceIds: []` — server
chặn trước khi gọi LLM, nên về rất nhanh (~0.1 s).

Code: `CadebotApiService.kt` → `suspend fun processQuery(message, history): AiMessage`

---

## Dependencies đã thêm

| Library | Version | Mục đích |
|---------|---------|----------|
| `com.squareup.okhttp3:okhttp` | `4.12.0` | HTTP client gọi Cadebot API |
| `android.permission.RECORD_AUDIO` | built-in | Quyền ghi âm |
| `android.permission.INTERNET` | built-in | Quyền gọi API |
| `android.media.MediaRecorder` | built-in | Ghi âm ra `.m4a` |
| `android.speech.tts.TextToSpeech` | built-in | Đọc `spokenText` |
| `org.json.JSONObject` | built-in | Parse response |

---

## Lưu ý bảo mật

- App không giữ credential nào — không có API key để lộ.
- `/chat` phía server hiện **không xác thực** và CORS để `*`. Chấp nhận được cho
  demo sau tên miền riêng tư; xem phần security notes trong
  [deployment.md](deployment.md).
- `local.properties` đã gitignore. Đừng commit nó, kể cả khi chỉ chứa URL.

---

## Lịch sử

Bản đầu dùng **Groq Whisper API** cho STT. Đã bỏ hoàn toàn: STT chuyển sang
PhoWhisper-large chạy trên chính Cadebot API, nên app còn đúng một backend và
không cần API key. Nếu gặp tàn dư nào còn nhắc `groq` trong code hay tài liệu
thì đó là sót, hãy xoá.
