// =============================變數/常數=============================\\
let isInCall = false;
let isMuted = false;
let ws = null;
let wsUrl = `ws://${window.location.host}/ws`;
let recorder = null;
let recordInterval = null;
let waveformAnimationId = null;
let lastWaveLevel = 0;
let systemPrompt = "你是一個帶有幽默感的語言模型，請用中文回答問題。";
let sessionId = null;
let heartbeatInterval = null;

const headerStatus = document.getElementById('header-status');
const statusText = document.getElementById('status-text');
const callBtn = document.getElementById('call-btn');
const muteBtn = document.getElementById('mute-btn');
const waveformEl = document.querySelector('.waveform');
const waveBars = waveformEl ? Array.from(waveformEl.querySelectorAll('.wave-bar')) : [];


const sampleRate = 24000;
const sampleBits = 16;
const numChannels = 1; // mono

// ============================心跳機制=============================\\
function startHeartbeat() {
    if (heartbeatInterval) return;
    heartbeatInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "request.ping" }));
        }
    }, 30000); // 每 30 秒發送一次心跳
}

function stopHeartbeat() {
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = null;
    }
}

// =============================錄音=============================\\
async function startRecording() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        throw new Error("ws not open");
    }

    await stopRecording();
    // 注意：必須設定 compiling: true 才能邊錄邊傳
    recorder = new Recorder({
        sampleBits: sampleBits,
        sampleRate: sampleRate,
        numChannels: numChannels,
        compiling: true        // 允許即時獲取 PCM 數據
    });

    try {
        // 請求麥克風權限並開始錄音
        await recorder.start();
        console.log("錄音已開始 (24kHz, 16-bit PCM)");

        startWaveformLoop();

        // 🌟 設定定時器，每 100 毫秒抓取一次新的 PCM 數據送給後端
        recordInterval = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN && !isMuted) {
                // getNextData() 回傳 PCM ArrayBuffer
                let pcmData = recorder.getNextData();
                if (pcmData && pcmData.byteLength > 0) {
                    console.log("送出 PCM 數據:", pcmData);
                    ws.send(pcmData);
                }
            }
        }, 100);

    } catch (error) {
        console.error("麥克風啟動失敗或被拒絕:", error);
        throw error;
    }
}

function stopRecording() {
    return new Promise((resolve) => {
        // 清除定時器
        if (recordInterval) {
            clearInterval(recordInterval);
            recordInterval = null;
        }

        stopWaveformLoop();

        // 停止並銷毀錄音實例，釋放記憶體
        if (recorder) {
            recorder.stop();
            recorder.destroy();
            recorder = null;
            console.log("錄音已停止");
        }
        resolve();
    });
}

const closeWs = () => {
    if (!ws) return;
    stopHeartbeat();
    try {
        ws.close();
    } catch { }
    ws = null;
    session = null;
};

const connect_ws = () => {
    return new Promise((resolve, reject) => {
        try {
            ws = new WebSocket(wsUrl);
        } catch (e) {
            ws = null;
            reject(e);
            return;
        }

        let settled = false;
        const timeout = window.setTimeout(() => {
            if (settled) return;
            settled = true;
            try {
                ws?.close();
            } catch { }
            ws = null;
            reject(new Error("ws timeout"));
        }, 6000);

        ws.onopen = () => {
            settled = true;
            ws.send(JSON.stringify({
                type: "request.session",
            }));
        };

        ws.onerror = () => {
            if (settled) return;
            settled = true;
            window.clearTimeout(timeout);
            reject(new Error("ws error"));
        };

        ws.onclose = () => {
            stopHeartbeat();
        };

        ws.onmessage = (event) => {
            let msg;
            try {
                msg = JSON.parse(event.data);
            } catch {
                console.warn("無法解析 ws 訊息", event.data);
                return;
            }
            // if (!msg || typeof msg.type !== "string") return;
            if (msg.type == "response.pong") {
                console.log("收到 pong");
                return;
            }

            if (msg.type === "response.error") {
                console.error("後端錯誤:", msg.error);
                alert("後端錯誤: " + msg.error);
                return;
            };

            if (msg.type === "response.session") {
                console.log("ws建立成功，session id:", msg.session_id);
                resolve(msg.session_id);
            };

            if (msg.type === "response.set_system_prompt") {
                console.log("系統提示詞設定成功");
            };

            if (msg.type === "response.asr_text") {
                // 如果是即時語音辨識的結果，就印出來
                if (msg.text) {
                    console.log(`辨識結果 [${msg.language}]:`, msg.text);
                    // 這裡可以把 msg.text 寫入到 HTML 的某個 <div> 裡讓使用者看到！
                    document.getElementById('transcript-text').innerText = msg.text;
                }
            };
        };
    });
};

// =============================UI/UX=============================\\
// 處理通話按鈕點擊
async function toggleCall() {
    isInCall = !isInCall;

    if (isInCall) {
        // 進入通話狀態
        headerStatus.classList.add('recording');
        statusText.innerText = '正在連線中...';
        connect_ws()
            .then((sid) => {
                console.log("WebSocket 連線成功，Session ID:", sid);
                sessionId = sid;
                ws.send(JSON.stringify({
                    type: "request.set_system_prompt",
                    session_id: sessionId,
                    system_prompt: systemPrompt
                }));
                startHeartbeat();
                headerStatus.classList.add('recording');
                statusText.innerText = '正在收音中...';
                callBtn.setAttribute('title', '結束通話');
                callBtn.setAttribute('aria-label', '結束通話');
                const icon = callBtn.querySelector('i');
                if (icon) {
                    icon.classList.remove('fa-phone');
                    icon.classList.add('fa-phone-slash');
                }
                callBtn.classList.add('in-call');
                startRecording();
            });

        // 模擬 LLM 思考狀態 (3秒後切換)
        // setTimeout(() => {
        //     if (isInCall) {
        //         statusText.innerText = 'LLM 思考中...';
        //     }
        // }, 3000);

    } else {
        // 結束通話狀態
        headerStatus.classList.remove('recording');
        callBtn.setAttribute('title', '開始通話');
        callBtn.setAttribute('aria-label', '開始通話');
        sessionId = null;
        closeWs();
        const icon = callBtn.querySelector('i');
        if (icon) {
            icon.classList.remove('fa-phone-slash');
            icon.classList.add('fa-phone');
        }
        callBtn.classList.remove('in-call');
        await stopRecording();
        muteBtn.setAttribute('title', '靜音');
        muteBtn.setAttribute('aria-label', '靜音');
        muteBtn.classList.remove('muted');
        setTimeout(() => {
            if (!isInCall) statusText.innerHTML = '準備就緒';
        }, 2000);
    }
}

// 處理靜音按鈕點擊
function toggleMute() {
    if (!isInCall) return; // 不在通話中點擊無效

    isMuted = !isMuted;
    if (isMuted) {
        muteBtn.setAttribute('title', '取消靜音');
        muteBtn.setAttribute('aria-label', '取消靜音');
        muteBtn.classList.add('muted');
        headerStatus.classList.remove('recording');
        statusText.innerText = '麥克風已靜音';
        stopWaveformLoop();
    } else {
        muteBtn.setAttribute('title', '靜音');
        muteBtn.setAttribute('aria-label', '靜音');
        muteBtn.classList.remove('muted');
        headerStatus.classList.add('recording');
        statusText.innerText = '正在收音中...';
        startWaveformLoop();
    }
}

// =============================等化器動畫=============================\\
function startWaveformLoop() {
    if (!waveBars.length) return;
    if (waveformAnimationId) return;

    const update = () => {
        if (!recorder || !isInCall || isMuted) {
            waveformAnimationId = null;
            resetWaveform();
            return;
        }

        const analyserData = recorder.getRecordAnalyseData && recorder.getRecordAnalyseData();
        const level = analyserData ? getWaveLevel(analyserData) : 0;
        const smoothed = lastWaveLevel * 0.7 + level * 0.3;
        lastWaveLevel = smoothed;

        renderWaveform(smoothed);
        waveformAnimationId = requestAnimationFrame(update);
    };

    waveformAnimationId = requestAnimationFrame(update);
}

function stopWaveformLoop() {
    if (waveformAnimationId) {
        cancelAnimationFrame(waveformAnimationId);
        waveformAnimationId = null;
    }
    lastWaveLevel = 0;
    resetWaveform();
}

function getWaveLevel(timeDomainData) {
    let sum = 0;
    for (let i = 0; i < timeDomainData.length; i++) {
        const centered = (timeDomainData[i] - 128) / 128;
        sum += centered * centered;
    }
    const rms = Math.sqrt(sum / timeDomainData.length);
    const boosted = Math.pow(rms * 3.2, 0.65);
    return Math.min(1, Math.max(0, boosted));
}

function renderWaveform(level) {
    if (!waveBars.length) return;

    const base = 0.08 + level * 1.25;
    const mid = (waveBars.length - 1) / 2;
    for (let i = 0; i < waveBars.length; i++) {
        const distance = Math.abs(i - mid) / mid;
        const envelope = 1 - Math.min(1, distance * 1.1);
        const shimmer = 0.9 + Math.sin((performance.now() / 140) + i) * 0.1;
        const scale = Math.min(1.6, base * (0.35 + envelope) * shimmer);
        waveBars[i].style.transform = `scaleY(${scale})`;
    }
}

function resetWaveform() {
    if (!waveBars.length) return;
    for (const bar of waveBars) {
        bar.style.transform = 'scaleY(0.15)';
    }
}