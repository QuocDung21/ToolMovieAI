#!/usr/bin/env python3
"""
ToolMovieAI - Colab GPU Worker Server
Cung cấp backend tăng tốc GPU qua FastAPI + PyCloudflared:
- Demucs v4 (Tách Vocal / BGM chất lượng phòng thu)
- Faster-Whisper Large-v3 (CUDA FP16 nhận diện tiếng Trung / đa ngôn ngữ siêu tốc)
"""

import os
import io
import sys
import uuid
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ToolMovieAI GPU Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = tempfile.gettempdir()
STORAGE_DIR = os.path.join(TEMP_DIR, "toolmovieai_colab")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Cache loaded Whisper models
WHISPER_MODELS = {}


def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            return {
                "available": True,
                "name": dev_name,
                "total_vram_gb": round(total_mem, 2),
                "free_vram_gb": round(free_mem, 2),
            }
    except Exception:
        pass
    return {"available": False, "name": "CPU", "total_vram_gb": 0, "free_vram_gb": 0}


def get_whisper_model(model_size: str = "large-v3"):
    from faster_whisper import WhisperModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    if model_size not in WHISPER_MODELS:
        print(f"🔥 Đang nạp mô hình Faster-Whisper {model_size} ({device} - {compute_type})...")
        WHISPER_MODELS[model_size] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return WHISPER_MODELS[model_size]


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@app.get("/")
@app.get("/health")
async def health():
    gpu = get_gpu_info()
    return {
        "status": "ready",
        "service": "ToolMovieAI Colab GPU Acceleration",
        "gpu": gpu,
        "supported_features": ["demucs_v4", "faster_whisper_large_v3", "batch_processing"],
    }


@app.post("/api/separate-bgm")
async def separate_bgm(file: UploadFile = File(...)):
    """
    Tách Vocal và BGM bằng Demucs v4 (htdemucs) trên CUDA.
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_path = os.path.join(session_dir, "input_audio.wav")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-d", device,
            "-o", session_dir,
            input_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Lỗi Demucs: {res.stderr}")

        model_out = os.path.join(session_dir, "htdemucs", "input_audio")
        vocal_src = os.path.join(model_out, "vocals.wav")
        bgm_src = os.path.join(model_out, "no_vocals.wav")

        final_vocal = os.path.join(session_dir, "vocal.wav")
        final_bgm = os.path.join(session_dir, "bgm.wav")
        shutil.copy2(vocal_src, final_vocal)
        shutil.copy2(bgm_src, final_bgm)

        return {
            "session_id": session_id,
            "vocal_url": f"/api/download/{session_id}/vocal.wav",
            "bgm_url": f"/api/download/{session_id}/bgm.wav",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_size: str = Form("large-v3"),
    language: str = Form("zh"),
    initial_prompt: Optional[str] = Form(None)
):
    """
    Nhận diện giọng nói đa ngôn ngữ bằng Faster-Whisper GPU.
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_path = os.path.join(session_dir, "audio.wav")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        model = get_whisper_model(model_size)
        prompt = initial_prompt or "以下是中文普通话对白，请使用简体中文输出字幕。" if language == "zh" else None

        segments, info = model.transcribe(
            input_path,
            language=language,
            initial_prompt=prompt,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=450),
        )

        srt_blocks = []
        seg_list = []
        idx = 1
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            start_str = format_timestamp(seg.start)
            end_str = format_timestamp(seg.end)
            srt_blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
            seg_list.append({"start": seg.start, "end": seg.end, "text": text})
            idx += 1

        srt_content = "\n".join(srt_blocks)
        return {
            "session_id": session_id,
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "srt_content": srt_content,
            "segments": seg_list,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-all")
async def process_all(
    file: UploadFile = File(...),
    model_size: str = Form("large-v3"),
    language: str = Form("zh"),
    separate_bgm: bool = Form(True),
):
    """
    Quy trình tích hợp 2 trong 1:
    1. Nhận audio thô từ Local Mac.
    2. Chạy Demucs v4 tách sạch BGM và Vocal trên GPU CUDA.
    3. Chạy Faster-Whisper Large-v3 nhận diện giọng nói từ vocal sạch.
    4. Trả về cả SRT và link tải BGM chất lượng cao.
    """
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_path = os.path.join(session_dir, "raw_audio.wav")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio_for_whisper = input_path
        has_bgm = False

        if separate_bgm:
            print("🎵 [Colab GPU] Đang chạy Demucs v4 tách Vocal & BGM...")
            cmd = [
                sys.executable, "-m", "demucs.separate",
                "-n", "htdemucs",
                "--two-stems", "vocals",
                "-d", device,
                "-o", session_dir,
                input_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                model_out = os.path.join(session_dir, "htdemucs", "raw_audio")
                vocal_src = os.path.join(model_out, "vocals.wav")
                bgm_src = os.path.join(model_out, "no_vocals.wav")
                final_vocal = os.path.join(session_dir, "vocal.wav")
                final_bgm = os.path.join(session_dir, "bgm.wav")
                shutil.copy2(vocal_src, final_vocal)
                shutil.copy2(bgm_src, final_bgm)
                audio_for_whisper = final_vocal
                has_bgm = True
            else:
                print(f"⚠️ Demucs cảnh báo: {res.stderr}")

        print(f"🎙️ [Colab GPU] Đang chạy Faster-Whisper ({model_size})...")
        model = get_whisper_model(model_size)
        prompt = "以下是中文普通话对白，请使用简体中文输出字幕。" if language == "zh" else None

        segments, info = model.transcribe(
            audio_for_whisper,
            language=language,
            initial_prompt=prompt,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=450),
        )

        srt_blocks = []
        idx = 1
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            start_str = format_timestamp(seg.start)
            end_str = format_timestamp(seg.end)
            srt_blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
            idx += 1

        srt_content = "\n".join(srt_blocks)

        return {
            "session_id": session_id,
            "status": "success",
            "detected_language": info.language,
            "duration": round(info.duration, 2),
            "srt_content": srt_content,
            "has_bgm": has_bgm,
            "bgm_url": f"/api/download/{session_id}/bgm.wav" if has_bgm else None,
            "vocal_url": f"/api/download/{session_id}/vocal.wav" if has_bgm else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    file_path = os.path.join(STORAGE_DIR, session_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File không tồn tại hoặc đã hết hạn")
    return FileResponse(file_path, filename=filename)


def run_tunnel(port: int = 8000):
    """Khởi chạy Cloudflare Tunnel miễn phí, không cần tài khoản hay token."""
    print("🌐 Đang khởi tạo Cloudflare Tunnel...")
    try:
        from pycloudflared import try_cloudflare
        tunnel_url = try_cloudflare(port=port)
        print("\n" + "=" * 65)
        print("🚀 TOOLMOVIEAI COLAB GPU WORKER ĐÃ HOÀNG TẤT KHỞI TẠO!")
        print(f"👉 COPY LINK NÀY DÁN VÀO WEB STUDIO TRÊN MAC:")
        print(f"   {tunnel_url}")
        print("=" * 65 + "\n")
        return tunnel_url
    except Exception as e:
        print(f"⚠️ Chưa có pycloudflared ({e}), chạy local port {port}.")
        return f"http://127.0.0.1:{port}"


if __name__ == "__main__":
    import uvicorn
    import threading

    port = 8000
    # In môi trường Colab, mở tunnel trong background thread
    threading.Thread(target=lambda: run_tunnel(port), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port)
