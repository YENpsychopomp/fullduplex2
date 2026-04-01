# record_and_predict.py 
import os
import time
import math
import urllib.request
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

import numpy as np
import pyaudio
from scipy.io import wavfile
import onnxruntime as ort

from inference import predict_endpoint

# --- 基础配置（固定 16 kHz 单声道，512 样本块）---
RATE = 16000
CHUNK = 512                     # Silero VAD 在 16 kHz 下需要 512 个样本
FORMAT = pyaudio.paInt16
CHANNELS = 1

# --- VAD 配置 ---
VAD_THRESHOLD = 0.5             # 语音概率阈值
PRE_SPEECH_MS = 200             # 触发前保留的毫秒数

# --- 动态端点检测配置 ---
EARLY_CHECK_MS = 80            # 静音后多久开始第一次检测
CHECK_INTERVAL_MS = 150         # 基础检测间隔
MIN_CHECK_INTERVAL_MS = 80      # 高置信度时的最小检测间隔
MAX_STOP_MS = 1500              # 最大静音等待时间（兜底）
MAX_DURATION_SECONDS = 8        # 每段音频的最大时长上限

# --- 置信度阈值 ---
HIGH_CONFIDENCE = 0.70          # 高置信度阈值，可立即结束
MEDIUM_CONFIDENCE = 0.50        # 中等置信度
LOW_CONFIDENCE = 0.30           # 低置信度，需继续等待

# --- 调试配置 ---
DEBUG_SAVE_WAV = False
TEMP_OUTPUT_WAV = "temp_output.wav"
DEBUG_LOG = True                # 是否打印检测日志

# --- Silero ONNX 模型 ---
ONNX_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
ONNX_MODEL_PATH = "silero_vad.onnx"
MODEL_RESET_STATES_TIME = 5.0


# class SileroVAD:
#     """Silero VAD ONNX 封装类，适用于 16 kHz 单声道，块大小为 512。"""

#     def __init__(self, model_path: str):
#         if ort is None:
#             raise RuntimeError("onnxruntime 尚未安裝，無法使用 SileroVAD。")
#         opts = ort.SessionOptions()
#         opts.inter_op_num_threads = 1
#         opts.intra_op_num_threads = 1
#         self.session = ort.InferenceSession(
#             model_path, providers=["CPUExecutionProvider"], sess_options=opts
#         )
#         self.context_size = 64
#         self._state = None
#         self._context = None
#         self._last_reset_time = time.time()
#         self._init_states()

#     def _init_states(self):
#         self._state = np.zeros((2, 1, 128), dtype=np.float32)
#         self._context = np.zeros((1, self.context_size), dtype=np.float32)

#     def maybe_reset(self):
#         if (time.time() - self._last_reset_time) >= MODEL_RESET_STATES_TIME:
#             self._init_states()
#             self._last_reset_time = time.time()

#     def prob(self, chunk_f32: np.ndarray) -> float:
#         """计算一个长度为 512 的音频块的语音概率。"""
#         x = np.reshape(chunk_f32, (1, -1))
#         if x.shape[1] != CHUNK:
#             raise ValueError(f"期望 {CHUNK} 个样本，实际得到 {x.shape[1]}")
#         x = np.concatenate((self._context, x), axis=1)

#         ort_inputs = {
#             "input": x.astype(np.float32),
#             "state": self._state,
#             "sr": np.array(16000, dtype=np.int64)
#         }
#         out, self._state = self.session.run(None, ort_inputs)
#         self._context = x[:, -self.context_size:]
#         self.maybe_reset()

#         return float(out[0][0])


class AsyncSmartTurnDetector:
    """异步 Smart Turn 检测器，支持预热和异步推理。"""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.current_future: Optional[Future] = None
        self.last_result: Optional[dict] = None
        self.last_audio_hash: Optional[int] = None
        self._lock = threading.Lock()

        # 预热模型（首次加载较慢）
        self._warmup()

    def _warmup(self):
        """预热模型，减少首次推理延迟。"""
        print("正在预热 Smart Turn 模型...")
        dummy_audio = np.zeros(RATE, dtype=np.float32)  # 1 秒静音
        predict_endpoint(dummy_audio)
        print("Smart Turn 模型预热完成。")

    def submit_async(self, audio_segment: np.ndarray) -> Future:
        """异步提交推理任务。"""
        with self._lock:
            # 取消之前的任务（如果还在运行）
            if self.current_future and not self.current_future.done():
                self.current_future.cancel()

            audio_hash = hash(audio_segment.tobytes()[-8000:])  # 只用最后 0.5 秒做哈希
            self.last_audio_hash = audio_hash
            self.current_future = self.executor.submit(self._run_inference, audio_segment, audio_hash)
            return self.current_future

    def _run_inference(self, audio_segment: np.ndarray, audio_hash: int) -> dict:
        """执行推理。"""
        t0 = time.perf_counter()
        result = predict_endpoint(audio_segment)
        result['inference_time_ms'] = (time.perf_counter() - t0) * 1000.0
        result['audio_hash'] = audio_hash

        with self._lock:
            self.last_result = result

        return result

    def get_result_if_ready(self) -> Optional[dict]:
        """非阻塞获取结果（如果已完成）。"""
        with self._lock:
            if self.current_future and self.current_future.done():
                try:
                    return self.current_future.result(timeout=0)
                except Exception:
                    return None
            return None

    def get_result_blocking(self, timeout: float = 0.5) -> Optional[dict]:
        """阻塞等待结果。"""
        with self._lock:
            future = self.current_future

        if future:
            try:
                return future.result(timeout=timeout)
            except Exception:
                return None
        return None

    def shutdown(self):
        """关闭线程池。"""
        self.executor.shutdown(wait=False)


class DynamicEndpointDetector:
    """动态端点检测器，实现智能静音判断策略。"""

    def __init__(self):
        self.chunk_ms = (CHUNK / RATE) * 1000.0

        # 转换为 chunk 数量
        self.early_check_chunks = math.ceil(EARLY_CHECK_MS / self.chunk_ms)
        self.base_interval_chunks = math.ceil(CHECK_INTERVAL_MS / self.chunk_ms)
        self.min_interval_chunks = math.ceil(MIN_CHECK_INTERVAL_MS / self.chunk_ms)
        self.max_stop_chunks = math.ceil(MAX_STOP_MS / self.chunk_ms)

        # 状态
        self.last_check_silence_chunks = 0
        self.last_probability = 0.0
        self.check_count = 0
        self.pending_inference = False

    def reset(self):
        """重置检测状态。"""
        self.last_check_silence_chunks = 0
        self.last_probability = 0.0
        self.check_count = 0
        self.pending_inference = False

    def get_dynamic_interval(self) -> int:
        """根据上次概率动态计算检测间隔（chunk 数量）。"""
        if self.last_probability >= HIGH_CONFIDENCE:
            # 高置信度：更频繁检测
            return self.min_interval_chunks
        elif self.last_probability >= MEDIUM_CONFIDENCE:
            # 中等置信度：正常间隔
            return self.base_interval_chunks
        else:
            # 低置信度：稍长间隔
            return int(self.base_interval_chunks * 1.5)

    def should_check(self, trailing_silence_chunks: int) -> bool:
        """判断是否应该进行端点检测。"""
        if self.pending_inference:
            return False

        # 首次检测
        if self.check_count == 0 and trailing_silence_chunks >= self.early_check_chunks:
            return True

        # 后续检测：基于动态间隔
        if self.check_count > 0:
            chunks_since_last = trailing_silence_chunks - self.last_check_silence_chunks
            interval = self.get_dynamic_interval()
            if chunks_since_last >= interval:
                return True

        return False

    def should_force_end(self, trailing_silence_chunks: int, since_trigger_chunks: int, max_chunks: int) -> bool:
        """判断是否强制结束。"""
        return (trailing_silence_chunks >= self.max_stop_chunks or
                since_trigger_chunks >= max_chunks)

    def on_check_started(self, trailing_silence_chunks: int):
        """记录检测开始。"""
        self.pending_inference = True

    def on_check_completed(self, trailing_silence_chunks: int, probability: float):
        """记录检测完成。"""
        self.last_check_silence_chunks = trailing_silence_chunks
        self.last_probability = probability
        self.check_count += 1
        self.pending_inference = False

    def should_end_by_confidence(self, probability: float, trailing_silence_chunks: int) -> bool:
        """基于置信度判断是否应该结束。"""
        silence_ms = trailing_silence_chunks * self.chunk_ms

        if probability >= HIGH_CONFIDENCE:
            # 高置信度：200ms 静音即可结束
            return silence_ms >= EARLY_CHECK_MS
        elif probability >= MEDIUM_CONFIDENCE:
            # 中等置信度：需要更多静音确认
            required_ms = EARLY_CHECK_MS + (HIGH_CONFIDENCE - probability) * 500
            return silence_ms >= required_ms
        else:
            # 低置信度：继续等待
            return False


class SileroVAD:
    """
    極簡版：單純使用 Silero VAD 判斷人聲，
    一旦偵測到講話後，停頓超過指定的秒數，就觸發結算。
    """
    def __init__(self, sample_rate: int = 24000, pause_sec: float = 2.0):
        self.sample_rate = sample_rate
        self.target_rate = 16000
        
        # 只需要載入 Silero VAD (請確保你有確保模型載入的邏輯，例如 vad.py 裡的 ensure_model)
        from vad import SileroVAD, ensure_model 
        self.vad = SileroVAD(ensure_model())
        
        # 緩衝區與 VAD 參數
        self._resample_buffer = np.array([], dtype=np.float32)
        self.CHUNK_SIZE = 512
        self.VAD_THRESHOLD = 0.5
        
        # 計算「停頓秒數」等同於幾個 chunk
        # 16kHz 下，512 個樣本 = 0.032 秒
        # 如果 pause_sec 是 2 秒，大約需要 62 個靜音 chunks
        self.max_silence_chunks = int((pause_sec * self.target_rate) / self.CHUNK_SIZE)
        
        self.speech_active = False
        self.trailing_silence_chunks = 0

    def _resample(self, wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return wav
        dur = wav.shape[0] / float(orig_sr)
        n_target = int(round(dur * target_sr))
        if n_target <= 0:
            return np.zeros((0,), dtype=np.float32)
        x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
        x_new = np.linspace(0.0, dur, num=n_target, endpoint=False)
        return np.interp(x_new, x_old, wav).astype(np.float32)

    def reset(self):
        """結算後重置狀態"""
        self.speech_active = False
        self.trailing_silence_chunks = 0
        self.vad.maybe_reset()

    def process_pcm16(self, audio_bytes: bytes) -> bool:
        if not audio_bytes:
            return False

        # 1. 轉成 16kHz float32
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_f32_24k = samples.astype(np.float32) / 32768.0
        audio_f32_16k = self._resample(audio_f32_24k, self.sample_rate, self.target_rate)
        
        self._resample_buffer = np.concatenate((self._resample_buffer, audio_f32_16k))

        # 2. 每次切 512 樣本給模型檢查
        while len(self._resample_buffer) >= self.CHUNK_SIZE:
            chunk = self._resample_buffer[:self.CHUNK_SIZE]
            self._resample_buffer = self._resample_buffer[self.CHUNK_SIZE:]
            
            # AI 判斷這 32ms 內有沒有人在講話
            is_speech = self.vad.prob(chunk) > self.VAD_THRESHOLD
            
            if not self.speech_active:
                if is_speech:
                    # 聽到聲音了，開始進入「講話中」狀態
                    self.speech_active = True
                    self.trailing_silence_chunks = 0
            else:
                if is_speech:
                    # 繼續講話中，靜音計時器歸零
                    self.trailing_silence_chunks = 0
                else:
                    # 沒講話，靜音計時器累加
                    self.trailing_silence_chunks += 1
                
                # 3. 如果靜音時間達到我們設定的秒數，就回傳 True 觸發結算！
                if self.trailing_silence_chunks >= self.max_silence_chunks:
                    return True 

        return False


def ensure_model(path: str = ONNX_MODEL_PATH, url: str = ONNX_MODEL_URL) -> str:
    if not os.path.exists(path):
        print("正在下载 Silero VAD ONNX 模型...")
        urllib.request.urlretrieve(url, path)
        print("ONNX 模型下载完成。")
    return path


def record_and_predict():
    """主录音和预测循环。"""
    if pyaudio is None:
        raise RuntimeError("pyaudio 尚未安裝，無法執行 record_and_predict。")
    if predict_endpoint is None:
        raise RuntimeError("inference.predict_endpoint 無法匯入，請確認 inference.py 可用。")

    # 计算派生参数
    chunk_ms = (CHUNK / RATE) * 1000.0
    pre_chunks = math.ceil(PRE_SPEECH_MS / chunk_ms)
    max_chunks = math.ceil(MAX_DURATION_SECONDS / (CHUNK / RATE))

    print(f"配置: 早期检测={EARLY_CHECK_MS}ms, 检测间隔={CHECK_INTERVAL_MS}ms, 最大静音={MAX_STOP_MS}ms")
    print(f"置信度阈值: 高={HIGH_CONFIDENCE}, 中={MEDIUM_CONFIDENCE}, 低={LOW_CONFIDENCE}")

    # 初始化组件
    vad = SileroVAD(ensure_model())
    detector = DynamicEndpointDetector()
    smart_turn = AsyncSmartTurnDetector()

    # 语音前环形缓冲区
    pre_buffer = deque(maxlen=pre_chunks)

    # 状态变量
    segment = []
    speech_active = False
    trailing_silence = 0
    since_trigger_chunks = 0

    # 初始化音频流
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("正在监听语音...（按 Ctrl+C 停止）")

    try:
        while True:
            # 读取音频块
            data = stream.read(CHUNK, exception_on_overflow=False)
            int16 = np.frombuffer(data, dtype=np.int16)
            f32 = (int16.astype(np.float32)) / 32768.0

            # VAD 检测
            is_speech = vad.prob(f32) > VAD_THRESHOLD

            if not speech_active:
                # 等待语音开始
                pre_buffer.append(f32)
                if is_speech:
                    # 触发：开始新的音频段
                    segment = list(pre_buffer)
                    segment.append(f32)
                    speech_active = True
                    trailing_silence = 0
                    since_trigger_chunks = 1
                    detector.reset()
                    print("[语音] 🎤 开始语音")
            else:
                # 已在音频段中
                segment.append(f32)
                since_trigger_chunks += 1

                if is_speech:
                    # 检测到语音：重置静音计数和检测状态
                    if trailing_silence > 0:
                        silence_duration_ms = trailing_silence * chunk_ms
                        print(f"[语音] 🔄 恢复语音（静音了 {silence_duration_ms:.0f} 毫秒）")
                        detector.reset()  # 语音恢复，重置检测状态
                    trailing_silence = 0
                else:
                    trailing_silence += 1

                # 检查是否强制结束
                if detector.should_force_end(trailing_silence, since_trigger_chunks, max_chunks):
                    stream.stop_stream()
                    audio_segment = np.concatenate(segment, dtype=np.float32)
                    final_silence_ms = trailing_silence * chunk_ms

                    # 同步获取最终结果
                    result = smart_turn.get_result_blocking(timeout=0.1)
                    if result is None:
                        result = predict_endpoint(audio_segment)

                    _process_segment(audio_segment, result, "强制结束", final_silence_ms)
                    _reset_state(segment, pre_buffer, detector)
                    speech_active = False
                    trailing_silence = 0
                    since_trigger_chunks = 0
                    stream.start_stream()
                    print("正在监听语音...")
                    continue

                # 检查是否应该进行端点检测
                if detector.should_check(trailing_silence):
                    detector.on_check_started(trailing_silence)

                    # 异步提交推理
                    audio_segment = np.concatenate(segment, dtype=np.float32)
                    smart_turn.submit_async(audio_segment.copy())

                # 检查异步推理结果
                result = smart_turn.get_result_if_ready()
                if result and detector.pending_inference:
                    prob = result.get("probability", 0)
                    inference_time = result.get("inference_time_ms", 0)
                    detector.on_check_completed(trailing_silence, prob)

                    if DEBUG_LOG:
                        silence_ms = trailing_silence * chunk_ms
                        print(f"[检测 #{detector.check_count}] 静音={silence_ms:.0f}ms, "
                              f"概率={prob:.3f}, 推理={inference_time:.1f}ms")

                    # 基于置信度判断是否结束
                    if detector.should_end_by_confidence(prob, trailing_silence):
                        stream.stop_stream()
                        audio_segment = np.concatenate(segment, dtype=np.float32)
                        final_silence_ms = trailing_silence * chunk_ms
                        _process_segment(audio_segment, result, "置信度判断", final_silence_ms)
                        _reset_state(segment, pre_buffer, detector)
                        speech_active = False
                        trailing_silence = 0
                        since_trigger_chunks = 0
                        stream.start_stream()
                        print("正在监听语音...")

    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        smart_turn.shutdown()


def _reset_state(segment: list, pre_buffer: deque, detector: DynamicEndpointDetector):
    """重置状态。"""
    segment.clear()
    pre_buffer.clear()
    detector.reset()


def _process_segment(segment_audio_f32: np.ndarray, result: dict, end_reason: str = "", silence_ms: float = 0):
    """处理完成的音频段。"""
    if segment_audio_f32.size == 0:
        print("捕获到空的音频段，跳过。")
        return

    if DEBUG_SAVE_WAV:
        if wavfile is None:
            raise RuntimeError("scipy 尚未安裝，無法輸出 debug wav。")
        wavfile.write(TEMP_OUTPUT_WAV, RATE, (segment_audio_f32 * 32767.0).astype(np.int16))

    dur_sec = segment_audio_f32.size / RATE
    pred = result.get("prediction", 0)
    prob = result.get("probability", float("nan"))
    inference_time = result.get("inference_time_ms", 0)

    print("=" * 40)
    print(f"音频段时长：{dur_sec:.2f} 秒")
    print(f"结束原因：{end_reason}（静音 {silence_ms:.0f} 毫秒）")
    print(f"预测结果：{'✅ 表达完整' if pred == 1 else '❌ 表达未完整'}")
    print(f"完整概率：{prob:.4f}")
    if inference_time > 0:
        print(f"推理耗时：{inference_time:.2f} 毫秒")
    print("=" * 40)


if __name__ == "__main__":
    record_and_predict()