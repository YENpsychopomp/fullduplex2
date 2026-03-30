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

load_dotenv()
app = FastAPI(title="full-duplex2")
logger = logging.getLogger("uvicorn.error")
session_manager = SessionManager()

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
    current_session_id = None
    asr_task = None # 用來追蹤 ASR 接收任務
    
    # 🌟 ASR Server 的 WebSocket 地址 (指向你的 WSL2)
    WS_ASR_URL = "ws://127.0.0.1:8001/ws/asr"
    
    try:
        # 🌟 建立與 ASR 伺服器的連線
        async with websockets.connect(WS_ASR_URL) as asr_ws:
            logger.info("成功連線到 ASR 伺服器！")
            
            # 🌟 定義一個背景任務：專門負責「不間斷地接收 ASR 回傳的文字」，並轉發給前端
            async def receive_asr_text():
                try:
                    while True:
                        asr_response_str = await asr_ws.recv()
                        asr_result = json.loads(asr_response_str)
                        
                        # 轉發 ASR 結果給前端 (讓前端顯示)
                        await websocket.send_text(json.dumps({
                            "type": "response.asr_text",
                            "text": asr_result.get("text", ""),
                            "language": asr_result.get("language", ""),
                            "status": asr_result.get("status", "")
                        }))
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("ASR 伺服器連線已關閉")
                except asyncio.CancelledError:
                    pass # 任務被正常取消
                except Exception as e:
                    logger.error(f"接收 ASR 文字發生錯誤: {e}")

            # 啟動背景任務
            asr_task = asyncio.create_task(receive_asr_text())

            # 這是原本負責接收前端訊息的主迴圈
            while True:
                # 1. 接收訊息 (前端掛斷時，這裡會直接觸發 WebSocketDisconnect 異常跳出迴圈)
                raw_msg = await websocket.receive()

                if raw_msg.get("type") == "websocket.disconnect":
                    logger.info("收到前端斷線訊號，準備結束通話...")
                    raise WebSocketDisconnect(code=raw_msg.get("code", 1000))
                
                # 2. 安全處理 JSON 訊息
                if "text" in raw_msg and raw_msg["text"]:
                    try:
                        data = json.loads(raw_msg["text"])
                    except json.JSONDecodeError:
                        logger.error("收到的不是有效的 JSON 格式")
                        continue # 解析失敗就跳過這次，繼續等下一筆
                    
                    # ==== 處理各種 request 指令 ====
                    if data.get("type") == "request.ping":
                        await websocket.send_text(json.dumps({
                            "type": "response.ping",
                            "msg": "pong"
                        }))
                        
                    elif data.get("type") == "request.session":
                        sid = session_manager.create_session()
                        current_session_id = sid
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
                    if current_session_id:
                        session_info = session_manager.get_session_info(current_session_id)
                        if session_info:
                            session_info.audio_buffer += audio_bytes

                    await asr_ws.send(audio_bytes)

    except ConnectionRefusedError:
        logger.error("無法連線到 ASR 伺服器！請確認 WSL2 中的 asr_server.py 是否已啟動。")
        await websocket.close(code=1011, reason="ASR Server Unavailable")
    except WebSocketDisconnect:
        logger.info("前端掛斷了，準備執行存檔...")
    except Exception as e:
        logger.error(f"WebSocket 發生未預期錯誤: {e}")
    finally:
        if asr_task:
            asr_task.cancel()
            
        if current_session_id:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "recorder"))
            saved_path = session_manager.save_session_audio(current_session_id, base_dir)
            
            if saved_path:
                logger.info(f"錄音已成功儲存: {saved_path}")
            else:
                logger.warning("錄音未儲存：原因可能是完全沒有收到音訊")
                
            session_manager.close_session(current_session_id)

app.mount("/", _HTTPOnlyStaticFiles(StaticFiles(directory="frontend", html=True)), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=7985, reload=True)