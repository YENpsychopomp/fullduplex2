 # full-duplex2

這個專案提供一個「前端即時錄音 + 後端轉送到 ASR + 即時回傳文字」的全雙工語音串流示範。
前端以 WebSocket 傳送 PCM Bytes，後端再轉送到 ASR 服務並把辨識結果回推給前端。

## 後端邏輯重點

- **FastAPI WebSocket `/ws`**：對前端唯一入口，負責接收控制訊息與音訊串流。
- **ASR 轉送**：後端會再連線到 `ws://127.0.0.1:8001/ws/asr`（ASR 服務）。
- **雙向串流**：
	- 前端音訊 (Bytes) -> 後端 -> ASR 服務。
	- ASR 回傳文字 -> 後端 -> 前端 `response.asr_text`。
- **Session**：`request.session` 建立 session id，後端用它管理緩衝音訊與結束存檔。
- **結束通話**：前端斷線後，後端會把收集到的 PCM 存在 `backend/recorder/`。

## 前端如何使用
- 在backend目錄下輸入
```bash
python main.py
```

前端的主要流程在 [frontend/js/all.js](frontend/js/all.js) 和 [frontend/js/recorder.js](frontend/js/recorder.js)。

1. 建立 WebSocket：`ws://<host>/ws`
2. 連線成功後送出 `request.session`
3. 收到 `response.session` 後可送出 `request.set_system_prompt`
4. 開始錄音，將 PCM Bytes 持續透過 `ws.send(pcmArrayBuffer)` 傳出
5. 後端回傳 `response.asr_text` 時，更新畫面上的即時轉錄
6. 結束通話時關閉 WebSocket，後端會自動存檔

## WebSocket 訊息規格

### 前端 -> 後端 (JSON)

- `request.ping`
```json
{ "type": "request.ping" }
```

- `request.session`
```json
{ "type": "request.session" }
```

- `request.set_system_prompt`
```json
{
	"type": "request.set_system_prompt",
	"session_id": "<session_id>",
	"system_prompt": "你是一個帶有幽默感的語言模型，請用中文回答問題。"
}
```

### 前端 -> 後端 (Bytes)

- **PCM 音訊串流**：直接送出 `ArrayBuffer`，格式為 24kHz / 16-bit / mono。

### 後端 -> 前端 (JSON)

- `response.ping`
```json
{ "type": "response.ping", "msg": "pong" }
```

- `response.session`
```json
{ "type": "response.session", "session_id": "<session_id>", "msg": "Session created" }
```

- `response.asr_text`
```json
{
	"type": "response.asr_text",
	"text": "辨識到的文字",
	"language": "zh",
	"status": "streaming"
}
```

## 前端常見動作對應

| 動作 | 前端送出的 type / payload | 後端回傳 | 前端使用方式 |
|---|---|---|---|
| 建立連線 | `request.session` | `response.session` | 取得 `session_id`，後續傳入設定
| 設定系統提示詞 | `request.set_system_prompt` | 無 (不回傳內容) | 本專案只存入 session，沒有 UI 變更
| 保持連線 | `request.ping` | `response.ping` | 記錄 log 或忽略
| 傳送音訊 | PCM Bytes | `response.asr_text` | 更新即時轉錄區塊
| 結束通話 | WebSocket close | 無 | 後端存檔到 `backend/recorder/`

## ASR 服務

ASR 服務在 [backend/qwan_example.py](backend/qwan_example.py)。
這個服務會接收 PCM Bytes，進行 24kHz -> 16kHz 重採樣後送入 Qwen3-ASR 模型，並以 WebSocket 回傳即時文字。

## 快速流程摘要

1. 前端連線 `/ws` -> 建立 session
2. 前端開始送 PCM Bytes
3. 後端把 Bytes 轉送到 ASR
4. ASR 回傳文字 -> 後端轉送前端
5. 前端更新 UI 的即時轉錄
6. 關閉 WebSocket -> 後端存檔音訊

## 後續開發
- 新增VAD
- 新增LLM回覆
- 新增TTS功能