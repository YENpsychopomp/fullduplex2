from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from dataclasses import dataclass, field
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydub import AudioSegment

import os
import asyncio
import json
import logging
import uuid
import requests
import uvicorn
import base64
import websockets

from sessions_manager import SessionManager, SessionData, AudioFormat
from vad import SmartStreamingVAD

load_dotenv()
app = FastAPI(title="full-duplex2")
logger = logging.getLogger("uvicorn.error")
session_manager = SessionManager()
PAUSE_FINISH_REASON = "pause_detected"
ASR_WS_CONNECT_TIMEOUT_SEC = 3


async def _send_finish_stream(asr_ws, reason: str):
    await asr_ws.send(json.dumps({
        "type": "request.finish_stream",
        "reason": reason,
    }))
    logger.info("已送出 ASR finish 指令，reason=%s", reason)

class _HTTPOnlyStaticFiles:
    def __init__(self, static_app: StaticFiles):
        self._static_app = static_app

    async def __call__(self, scope, receive, send):
        """
        對 StaticFiles 進行封裝，以確保只處理 HTTP 請求。
        """
        if scope.get("type") != "http":
            if scope.get("type") == "websocket":
                await send({"type": "websocket.close", "code": 1000})
            return
        await self._static_app(scope, receive, send)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("前端 WebSocket 已連線")
    current_session_id = None
    asr_task = None # 用來追蹤 ASR 接收任務
    asr_ws = None
    pause_vad = SmartStreamingVAD(sample_rate=24000)
    
    # 🌟 ASR Server 的 WebSocket 地址 (指向你的 WSL2)
    WS_ASR_URL = "ws://127.0.0.1:8001/ws/asr"
    
    async def _close_asr_connection(reason: str):
        nonlocal asr_task, asr_ws
        if asr_task:
            asr_task.cancel()
            logger.info("ASR 接收背景任務已取消，reason=%s", reason)
            asr_task = None
        if asr_ws:
            try:
                await asr_ws.close()
            except Exception:
                pass
            asr_ws = None
            logger.info("ASR 連線已關閉，reason=%s", reason)

    async def _receive_asr_text(active_asr_ws):
        nonlocal asr_ws
        try:
            while True:
                asr_response_str = await active_asr_ws.recv()
                asr_result = json.loads(asr_response_str)
                status = asr_result.get("status", "")
                asr_text = asr_result.get("text", "")
                asr_language = asr_result.get("language", "")
                logger.info("ASR 回傳 status=%s, language=%s, text=%s", status, asr_language, asr_text)

                await websocket.send_text(json.dumps({
                    "type": "response.asr_text",
                    "text": asr_text,
                    "language": asr_language,
                    "status": status
                }))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("ASR 伺服器連線已關閉")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("接收 ASR 文字發生錯誤: %s", e)
        finally:
            if asr_ws is active_asr_ws:
                asr_ws = None

    async def _ensure_asr_connection() -> bool:
        nonlocal asr_ws, asr_task
        if asr_ws and not asr_ws.closed:
            return True

        try:
            asr_ws = await asyncio.wait_for(
                websockets.connect(WS_ASR_URL),
                timeout=ASR_WS_CONNECT_TIMEOUT_SEC,
            )
            logger.info("成功連線到 ASR 伺服器")
            asr_task = asyncio.create_task(_receive_asr_text(asr_ws))
            logger.info("ASR 接收背景任務已啟動")
            return True
        except Exception as e:
            asr_ws = None
            logger.warning("ASR 連線失敗，稍後重試: %s", e)
            return False

    try:
        # 啟動時先嘗試連一次，不成功也不讓整個前端流程中斷
        await _ensure_asr_connection()

        # 主迴圈：永遠以處理前端訊息為主
        while True:
            raw_msg = await websocket.receive()
            logger.info("收到前端訊息 type=%s", raw_msg.get("type"))

            if raw_msg.get("type") == "websocket.disconnect":
                logger.info("收到前端斷線訊號，準備結束通話...")
                raise WebSocketDisconnect(code=raw_msg.get("code", 1000))

            # 2. 安全處理 JSON 訊息
            if "text" in raw_msg and raw_msg["text"]:
                try:
                    data = json.loads(raw_msg["text"])
                    logger.info("收到前端 JSON 指令 type=%s", data.get("type"))
                except json.JSONDecodeError:
                    logger.error("收到的不是有效的 JSON 格式")
                    continue # 解析失敗就跳過這次，繼續等下一筆

                # ==== 處理各種 request 指令 ====
                if data.get("type") == "request.ping":
                    logger.info("處理 request.ping")
                    await websocket.send_text(json.dumps({
                        "type": "response.ping",
                        "msg": "pong"
                    }))

                elif data.get("type") == "request.session":
                    sid = session_manager.create_session()
                    current_session_id = sid
                    pause_vad.reset()
                    logger.info("建立 session 成功 sid=%s", sid)
                    await websocket.send_text(json.dumps({
                        "type": "response.session",
                        "session_id": sid,
                        "msg": f"Session created with ID: {sid}"
                    }))

                elif data.get("type") == "request.set_system_prompt":
                    sid = data.get("session_id")
                    session_info = session_manager.get_session_info(sid)
                    if session_info:
                        session_info.system_prompt = data.get("system_prompt")
                        logger.info("更新 system prompt sid=%s", sid)
                    else:
                        logger.info("set_system_prompt 找不到 session sid=%s", sid)
                            
                    # elif data.get("type") == "request.audio_data":
                    #     sid = data.get("session_id")
                    #     audio_format = data.get("audio_format", {})
                    #     session_info = session_manager.get_session_info(sid)
                        
                    #     if session_info:
                    #         current_session_id = sid
                    #         session_info.audio_formate.format = audio_format.get("format", "pcm")
                    #         session_info.audio_formate.sample_rate = audio_format.get("sample_rate", 24000)
                    #         session_info.audio_formate.sample_width = audio_format.get("sample_width", audio_format.get("sample_bits", 16) // 8)
                    #         session_info.audio_formate.channels = audio_format.get("channels", 1)
                    #         session_info.audio_formate.has_set = True
                            
                    #         audio_data_b64 = data.get("audio_data")
                    #         if audio_data_b64:
                    #             try:
                    #                 # 安全地解碼並累加音訊
                    #                 audio_bytes = base64.b64decode(audio_data_b64)
                    #                 session_info.audio_buffer += audio_bytes
                                    
                    #                 # 🌟 核心新增：將解碼後的純音訊 Bytes 丟給 ASR 伺服器
                    #                 await asr_ws.send(audio_bytes)
                                    
                    #             except Exception as e:
                    #                 logger.error(f"Base64 音訊解碼或傳送失敗: {e}")
                    #     else:
                    #         await websocket.send_text(json.dumps({
                    #             "type": "response.audio_data",
                    #             "session_id": sid,
                    #             "msg": f"Session {sid} not found"
                    #         }))
                            
            elif "bytes" in raw_msg and raw_msg["bytes"]:
                audio_bytes = raw_msg["bytes"]
                logger.info("收到音訊 bytes=%d", len(audio_bytes))
                if current_session_id:
                    session_info = session_manager.get_session_info(current_session_id)
                    if session_info:
                        session_info.audio_buffer += audio_bytes
                        logger.info("音訊已累加到 session sid=%s, total_bytes=%d", current_session_id, len(session_info.audio_buffer))

                connected = await _ensure_asr_connection()
                if connected and asr_ws:
                    try:
                        await asr_ws.send(audio_bytes)
                        logger.info("音訊已轉送至 ASR")

                        if pause_vad.process_pcm16(audio_bytes):
                            logger.info("VAD 偵測到停頓，準備要求 ASR finish")
                            await _send_finish_stream(asr_ws, PAUSE_FINISH_REASON)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("送音訊到 ASR 失敗，連線已中斷，將在下個 chunk 重連")
                        await _close_asr_connection("send_audio_connection_closed")
                else:
                    logger.info("ASR 暫不可用，僅保留音訊於 session buffer")
            else:
                logger.info("收到未處理訊息內容，略過")

    except WebSocketDisconnect:
        logger.info("前端掛斷了，準備執行存檔...")
    except Exception as e:
        logger.exception("WebSocket 發生未預期錯誤: %s", e)
    finally:
        await _close_asr_connection("frontend_disconnected")
            
        if current_session_id:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "recorder"))
            saved_path = session_manager.save_session_audio(current_session_id, base_dir)
            
            if saved_path:
                logger.info(f"錄音已成功儲存: {saved_path}")
            else:
                logger.warning("錄音未儲存：原因可能是完全沒有收到音訊")
                
            session_manager.close_session(current_session_id)
            logger.info("session 已關閉 sid=%s", current_session_id)

app.mount("/", _HTTPOnlyStaticFiles(StaticFiles(directory="frontend", html=True)), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=7985, reload=True)