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
import logging
import time
import re
import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("toolmovieai-worker")

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


def get_device_allocation():
    """Phân bổ GPU tối ưu: Nếu có 2 GPU (như Kaggle T4 x2), tách riêng Demucs và Whisper."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu", "cpu", 0
        count = torch.cuda.device_count()
        if count >= 2:
            # GPU 0: Demucs (tách nhạc nền), GPU 1: Whisper (nhận diện phụ đề)
            return "cuda:0", "cuda:1", count
        return "cuda:0", "cuda:0", count
    except Exception:
        return "cpu", "cpu", 0


def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            total_mem = sum(torch.cuda.get_device_properties(i).total_memory for i in range(count)) / (1024 ** 3)
            free_mem = sum(torch.cuda.mem_get_info(i)[0] for i in range(count)) / (1024 ** 3)
            gpu_desc = f"{count}x {names[0]}" if count > 1 else names[0]
            return {
                "available": True,
                "name": gpu_desc,
                "device_count": count,
                "total_vram_gb": round(total_mem, 2),
                "free_vram_gb": round(free_mem, 2),
            }
    except Exception:
        pass
    return {"available": False, "name": "CPU", "device_count": 0, "total_vram_gb": 0, "free_vram_gb": 0}


def get_whisper_model(model_size: str = "large-v3"):
    from faster_whisper import WhisperModel
    import torch

    _, whisper_device, _ = get_device_allocation()
    compute_type = "float16" if "cuda" in whisper_device else "int8"
    cache_key = f"{model_size}_{whisper_device}"

    if cache_key not in WHISPER_MODELS:
        logger.info("MODEL loading Faster-Whisper %s (%s - %s)", model_size, whisper_device, compute_type)
        device_type = "cuda" if "cuda" in whisper_device else "cpu"
        device_index = int(whisper_device.split(":")[1]) if ":" in whisper_device else 0
        WHISPER_MODELS[cache_key] = WhisperModel(
            model_size,
            device=device_type,
            device_index=device_index,
            compute_type=compute_type
        )
        logger.info("MODEL ready Faster-Whisper %s", model_size)
    else:
        logger.info("MODEL cache hit Faster-Whisper %s", model_size)
    return WHISPER_MODELS[cache_key]


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
    logger.info("HEALTH status=ready gpu=%s", gpu.get("name", "unknown"))
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
    started = time.monotonic()
    logger.info("REQUEST separate-bgm session=%s filename=%s", session_id, file.filename or "audio")
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_path = os.path.join(session_dir, "input_audio.wav")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        import torch
        demucs_device, _, _ = get_device_allocation()
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-d", demucs_device,
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
        logger.exception("ERROR separate-bgm session=%s", session_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("REQUEST complete separate-bgm session=%s elapsed_ms=%d", session_id, round((time.monotonic() - started) * 1000))


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
    started = time.monotonic()
    logger.info(
        "REQUEST transcribe session=%s filename=%s model=%s language=%s",
        session_id,
        file.filename or "audio",
        model_size,
        language,
    )
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_path = os.path.join(session_dir, "audio.wav")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        model = get_whisper_model(model_size)
        logger.info("TRANSCRIBE start session=%s", session_id)
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
        logger.info("TRANSCRIBE complete session=%s segments=%d duration=%.2fs", session_id, len(seg_list), info.duration)
        return {
            "session_id": session_id,
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "srt_content": srt_content,
            "segments": seg_list,
        }
    except Exception as e:
        logger.exception("ERROR transcribe session=%s", session_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("REQUEST complete transcribe session=%s elapsed_ms=%d", session_id, round((time.monotonic() - started) * 1000))


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
    started = time.monotonic()
    logger.info(
        "REQUEST process-all session=%s filename=%s model=%s separate_bgm=%s",
        session_id,
        file.filename or "audio",
        model_size,
        separate_bgm,
    )
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    input_path = os.path.join(session_dir, "raw_audio.wav")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        import torch
        demucs_device, _, _ = get_device_allocation()
        audio_for_whisper = input_path
        has_bgm = False

        if separate_bgm:
            logger.info("DEMUCS start session=%s device=%s", session_id, demucs_device)
            cmd = [
                sys.executable, "-m", "demucs.separate",
                "-n", "htdemucs",
                "--two-stems", "vocals",
                "-d", demucs_device,
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
                logger.info("DEMUCS complete session=%s", session_id)
            else:
                logger.warning("DEMUCS failed session=%s detail=%s", session_id, res.stderr[-500:])

        logger.info("WHISPER start session=%s model=%s", session_id, model_size)
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
        logger.info("WHISPER complete session=%s segments=%d duration=%.2fs", session_id, idx - 1, info.duration)

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
        logger.exception("ERROR process-all session=%s", session_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("REQUEST complete process-all session=%s elapsed_ms=%d", session_id, round((time.monotonic() - started) * 1000))


@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    file_path = os.path.join(STORAGE_DIR, session_id, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File không tồn tại hoặc đã hết hạn")
    return FileResponse(file_path, filename=filename)


# =====================================================================
# DUBBING PIPELINE ON COLAB
# =====================================================================

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

POSITIONS = {
    "top-right": "W-w-15:15",
    "top-left": "15:15",
    "bottom-right": "W-w-15:H-h-15",
    "bottom-left": "15:H-h-15",
    "center": "(W-w)/2:(H-h)/2",
}


def sanitize_tts_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'<[^>]+>', '', text)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def generate_silence_wav(duration: float, output_wav: str):
    dur = max(0.2, duration)
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{dur:.3f}",
        "-ar", "44100", "-ac", "2",
        output_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def generate_speech_for_segment(
    text: str,
    voice: str,
    target_duration: float,
    output_wav: str,
    rate: str = "+0%",
    pitch: str = "+0Hz"
):
    import edge_tts
    t_clean = sanitize_tts_text(text)
    if not t_clean or not any(c.isalnum() for c in t_clean):
        generate_silence_wav(target_duration, output_wav)
        return

    candidates = [t_clean]
    if t_clean.endswith('.'):
        candidates.append(t_clean[:-1].strip())
    elif t_clean.endswith(('?', '!', '…', ':')):
        candidates.append(t_clean[:-1].strip() + '.')
    else:
        candidates.append(t_clean + '.')
        candidates.append(t_clean + '!')

    fallback_voice = "vi-VN-NamMinhNeural" if "HoaiMy" in voice else "vi-VN-HoaiMyNeural"
    voices_to_try = [voice, fallback_voice]

    temp_mp3 = output_wav + ".temp.mp3"
    success = False
    last_error = None

    for current_voice in voices_to_try:
        if success:
            break
        for c in candidates:
            if success:
                break
            for attempt in range(1, 3):
                try:
                    comm = edge_tts.Communicate(c, current_voice, rate=rate, pitch=pitch)
                    await comm.save(temp_mp3)
                    if os.path.exists(temp_mp3) and os.path.getsize(temp_mp3) > 0:
                        success = True
                        break
                    else:
                        raise RuntimeError("Edge-TTS returned an empty audio file")
                except Exception as error:
                    last_error = error
                    if os.path.exists(temp_mp3):
                        try:
                            os.remove(temp_mp3)
                        except OSError:
                            pass
                    await asyncio.sleep(0.5 * attempt)

    if not success or not os.path.exists(temp_mp3) or os.path.getsize(temp_mp3) == 0:
        logger.warning("TTS fallback to silence for segment text=%r error=%s", text[:50], last_error)
        generate_silence_wav(target_duration, output_wav)
        return

    raw_dur = get_media_duration(temp_mp3)
    if raw_dur <= 0:
        raw_dur = target_duration

    ratio = raw_dur / target_duration
    if ratio > 1.05:
        speed = min(1.38, ratio)
        cmd = [
            FFMPEG, "-y", "-i", temp_mp3,
            "-filter:a", f"atempo={speed:.3f}",
            "-ar", "44100", "-ac", "2",
            output_wav
        ]
    else:
        cmd = [
            FFMPEG, "-y", "-i", temp_mp3,
            "-ar", "44100", "-ac", "2",
            output_wav
        ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.exists(output_wav) or os.path.getsize(output_wav) == 0:
        generate_silence_wav(target_duration, output_wav)

    if os.path.exists(temp_mp3):
        try:
            os.remove(temp_mp3)
        except OSError:
            pass


def get_media_duration(file_path: str) -> float:
    cmd = [FFMPEG, "-i", file_path]
    res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', res.stderr)
    if m:
        h, mn, s = m.groups()
        return int(h) * 3600 + int(mn) * 60 + float(s)
    return 0.0


def parse_srt_text(content: str) -> list:
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []
    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            continue
        time_idx = 1 if lines[0].isdigit() else 0
        if time_idx >= len(lines) or "-->" not in lines[time_idx]:
            continue
        time_line = lines[time_idx]
        text = " ".join(lines[time_idx + 1:]).strip()
        if not text:
            continue
        m = re.match(r'(\d+):(\d+):(\d+)[,\.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,\.](\d+)', time_line)
        if not m:
            continue
        sh, sm, ss, sms = map(int, m.groups()[:4])
        eh, em, es, ems = map(int, m.groups()[4:])
        start_sec = sh * 3600 + sm * 60 + ss + sms / 1000.0
        end_sec = eh * 3600 + em * 60 + es + ems / 1000.0
        dur_sec = max(0.2, end_sec - start_sec)
        subtitles.append({
            "start": start_sec,
            "end": end_sec,
            "duration": dur_sec,
            "text": text
        })
    return subtitles


@app.post("/api/dub-video")
async def dub_video(
    file: UploadFile = File(...),
    srt_content: str = Form(...),
    voice: str = Form("vi-VN-NamMinhNeural"),
    rate: str = Form("+0%"),
    pitch: str = Form("+0Hz"),
    bgm_volume: float = Form(0.30),
    separate_bgm: bool = Form(True),
    hardsub: bool = Form(True),
    logo_file: Optional[UploadFile] = File(None),
    logo_pos: str = Form("top-right"),
    logo_width: int = Form(150),
    session_id: Optional[str] = Form(None),
):
    """
    Quy trình Lồng tiếng Video & Xuất File hoàn chỉnh trên Cloud GPU:
    1. Nhận video, phụ đề tiếng Việt và các cấu hình.
    2. Tách/lấy nhạc nền BGM bằng Demucs v4 trên CUDA.
    3. Tạo giọng đọc AI Edge-TTS tiếng Việt (với cơ chế chống lỗi, fallback).
    4. Trộn giọng + BGM và căn chỉnh mốc thời gian bằng numpy & soundfile.
    5. Render hoàn chỉnh video (in hardsub + logo) bằng GPU Nvidia (h264_nvenc).
    """
    if not session_id or not os.path.exists(os.path.join(STORAGE_DIR, session_id)):
        session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    started = time.monotonic()

    logger.info(
        "REQUEST dub-video session=%s filename=%s voice=%s separate_bgm=%s hardsub=%s",
        session_id,
        file.filename or "video.mp4",
        voice,
        separate_bgm,
        hardsub,
    )

    video_path = os.path.join(session_dir, "input_video.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    srt_path = os.path.join(session_dir, "vi_sub.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    logo_path = None
    if logo_file and logo_file.filename:
        logo_path = os.path.join(session_dir, "logo.png")
        with open(logo_path, "wb") as f:
            shutil.copyfileobj(logo_file.file, f)

    try:
        total_duration = get_media_duration(video_path)
        subs = parse_srt_text(srt_content)
        if not subs:
            raise HTTPException(status_code=400, detail="File phụ đề không hợp lệ hoặc không có câu thoại nào!")

        logger.info("TTS generating %d segments on Colab...", len(subs))
        seg_files = []
        for idx, sub in enumerate(subs, start=1):
            seg_wav = os.path.join(session_dir, f"seg_{idx:04d}.wav")
            await generate_speech_for_segment(
                sub["text"], voice, sub["duration"], seg_wav, rate=rate, pitch=pitch
            )
            seg_files.append((sub["start"], seg_wav))
            await asyncio.sleep(0.25)

        logger.info("ALIGN timeline voice track...")
        import numpy as np
        import soundfile as sf

        sr = 44100
        total_samples = int(max(total_duration + 5.0, subs[-1]["end"] + 5.0) * sr)
        timeline = np.zeros((total_samples, 2), dtype=np.float32)

        for start_sec, wav_file in seg_files:
            if not os.path.exists(wav_file):
                continue
            data, file_sr = sf.read(wav_file, dtype="float32", always_2d=True)
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            start_idx = int(round(start_sec * sr))
            end_idx = min(total_samples, start_idx + len(data))
            seg_len = end_idx - start_idx
            if seg_len > 0:
                timeline[start_idx:end_idx] += data[:seg_len]

        timeline = np.clip(timeline, -0.98, 0.98)
        voice_track = os.path.join(session_dir, "voice_track.wav")
        sf.write(voice_track, timeline[:int(total_duration * sr)], sr)

        # Xử lý BGM
        demucs_device, _, _ = get_device_allocation()
        bgm_track = os.path.join(session_dir, "bgm_scaled.wav")
        clean_bgm = os.path.join(session_dir, "bgm.wav")

        if separate_bgm:
            if not os.path.exists(clean_bgm):
                logger.info("DEMUCS start for dubbing session=%s device=%s", session_id, demucs_device)
                cmd = [
                    sys.executable, "-m", "demucs.separate",
                    "-n", "htdemucs",
                    "--two-stems", "vocals",
                    "-d", demucs_device,
                    "-o", session_dir,
                    video_path
                ]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0:
                    model_out = os.path.join(session_dir, "htdemucs", "input_video")
                    bgm_src = os.path.join(model_out, "no_vocals.wav")
                    if os.path.exists(bgm_src):
                        shutil.copy2(bgm_src, clean_bgm)

            source_for_bgm = clean_bgm if os.path.exists(clean_bgm) else video_path
        else:
            source_for_bgm = video_path

        subprocess.run([
            FFMPEG, "-y", "-i", source_for_bgm,
            "-vn", "-filter:a", f"volume={bgm_volume:.2f}",
            "-ar", "44100", "-ac", "2", bgm_track
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Trộn giọng + BGM
        mixed_audio = os.path.join(session_dir, "mixed_audio.wav")
        subprocess.run([
            FFMPEG, "-y",
            "-i", voice_track,
            "-i", bgm_track,
            "-filter_complex", "[0:a]volume=1.15[v];[1:a]volume=1.0[bg];[v][bg]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:level=0[aout]",
            "-map", "[aout]",
            mixed_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Xuất video hoàn chỉnh
        out_video = os.path.join(session_dir, "dubbed_video.mp4")
        pos_expr = POSITIONS.get(logo_pos, POSITIONS["top-right"])

        def check_filter(fname):
            try:
                out = subprocess.run([FFMPEG, "-filters"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                return bool(re.search(rf"\b{fname}\b", out.stdout))
            except Exception:
                return False

        can_hardsub = check_filter("subtitles") or check_filter("ass")

        inputs = ["-i", video_path]
        filter_parts = []
        current_v = "0:v"

        if hardsub:
            if can_hardsub:
                escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
                filter_parts.append(f"[{current_v}]subtitles=filename='{escaped_srt}'[vsub]")
                current_v = "vsub"
            else:
                logger.warning("FFmpeg missing subtitles filter")

        if logo_path and os.path.exists(logo_path):
            inputs.extend(["-i", logo_path])
            logo_idx = 1
            filter_parts.append(f"[{logo_idx}:v]scale={logo_width}:-1[wm]")
            filter_parts.append(f"[{current_v}][wm]overlay={pos_expr}[vout]")
            current_v = "vout"

        audio_input_idx = len(inputs) // 2
        inputs.extend(["-i", mixed_audio])

        cmd = [FFMPEG, "-y"] + inputs
        encoder = "h264_nvenc" if "cuda" in demucs_device else "libx264"
        if filter_parts:
            cmd.extend([
                "-filter_complex", ";".join(filter_parts),
                "-map", f"[{current_v}]",
                "-map", f"{audio_input_idx}:a",
                "-c:v", encoder,
                "-b:v", "2500k",
            ])
        else:
            cmd.extend([
                "-map", "0:v",
                "-map", f"{audio_input_idx}:a",
                "-c:v", "copy",
            ])

        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest", out_video])

        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 and encoder == "h264_nvenc":
            cmd[cmd.index("h264_nvenc")] = "libx264"
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        if res.returncode != 0:
            logger.warning("Render failed with filters, fallback to copy: %s", res.stderr[-200:])
            fallback_cmd = [
                FFMPEG, "-y",
                "-i", video_path,
                "-i", mixed_audio,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                out_video
            ]
            fb_res = subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if fb_res.returncode != 0:
                raise RuntimeError(f"FFmpeg render thất bại: {fb_res.stderr[-300:]}")

        logger.info("DUB-VIDEO complete session=%s duration=%.2fs", session_id, total_duration)
        return {
            "status": "success",
            "session_id": session_id,
            "filename": "dubbed_video.mp4",
            "download_url": f"/api/download/{session_id}/dubbed_video.mp4"
        }
    except Exception as e:
        logger.exception("ERROR dub-video session=%s", session_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("REQUEST complete dub-video session=%s elapsed_ms=%d", session_id, round((time.monotonic() - started) * 1000))


def run_tunnel(port: int = 8000):
    """Khởi chạy Cloudflare Tunnel miễn phí, không cần tài khoản hay token."""
    logger.info("TUNNEL starting port=%d", port)
    try:
        from pycloudflared import try_cloudflare
        tunnel_url = try_cloudflare(port=port)
        logger.info("TUNNEL ready url=%s", tunnel_url)
        return tunnel_url
    except Exception as e:
        logger.exception("TUNNEL failed; falling back to local port=%d", port)
        return f"http://127.0.0.1:{port}"


if __name__ == "__main__":
    import uvicorn
    import threading

    port = 8000
    # In môi trường Colab, mở tunnel trong background thread
    logger.info("SERVER starting host=0.0.0.0 port=%d", port)
    threading.Thread(target=lambda: run_tunnel(port), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port)
