#!/usr/bin/env python3
"""
Tool trích xuất phụ đề tiếng Trung từ Video (Speech-to-Text) & Tách Nhạc Nền
Kế thừa trực tiếp mô hình Faster-Whisper và Sherpa-ONNX Spleeter từ pyvideotrans core.
"""

import os
import sys
import json
import uuid
import shutil
import argparse
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import tempfile
import time
import threading
import math
import wave
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PYVIDEOTRANS_DIR = PROJECT_DIR / "pyvideotrans"
def get_ffmpeg_binary() -> str:
    candidates = [
        str(PYVIDEOTRANS_DIR / "ffmpeg" / "ffmpeg"),
        "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
        "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
        shutil.which("ffmpeg-full"),
        shutil.which("ffmpeg"),
        "ffmpeg",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return shutil.which("ffmpeg") or "ffmpeg"

FFMPEG = get_ffmpeg_binary()


def clean_colab_url(colab_url: str) -> str:
    """Chuẩn hóa đường dẫn Colab URL: loại bỏ khoảng trắng, dấu gạch chéo cuối, thêm https:// nếu thiếu."""
    if not colab_url:
        return ""
    u = colab_url.strip().rstrip("/")
    if not (u.startswith("http://") or u.startswith("https://")):
        u = f"https://{u}"
    return u


def check_colab_worker(colab_url: str, max_retries: int = 3, timeout: int = 25) -> dict:
    """Kiểm tra tình trạng máy chủ GPU Colab với cơ chế thử lại cho Cloudflare Quick Tunnels."""
    clean_url = clean_colab_url(colab_url)
    if not clean_url:
        raise ValueError("URL máy chủ Colab không hợp lệ hoặc đang để trống!")

    url = f"{clean_url}/health"
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ToolMovieAI-Client"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.reason}"
            if e.code not in (502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = str(e)

        if attempt < max_retries:
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"Không thể kết nối máy chủ Colab GPU ({clean_url}): {last_err}")


def test_colab_ai(colab_url: str, model_size: str = "tiny") -> dict:
    """Run a small transcription request to verify cloud inference end-to-end."""
    clean_url = clean_colab_url(colab_url)
    check_colab_worker(clean_url)
    sample_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as sample:
            sample_path = sample.name
        with wave.open(sample_path, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            # Use a quiet tone instead of a zero-length/silent payload. Some
            # Whisper/FFmpeg builds reject fully silent synthetic input.
            frames = bytearray()
            for index in range(32_000):
                sample = int(800 * math.sin(2 * math.pi * 440 * index / 16_000))
                frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
            audio.writeframes(frames)

        started = time.monotonic()
        result = upload_file_multipart(
            f"{clean_url}/api/transcribe",
            sample_path,
            {"model_size": model_size, "language": "zh"},
            timeout=180,
        )
        return {
            "status": "ok",
            "latency_ms": round((time.monotonic() - started) * 1000),
            "model_size": model_size,
            "detected_language": result.get("detected_language"),
            "duration": result.get("duration", 0),
            "segments": len(result.get("segments", [])),
        }
    finally:
        if sample_path and os.path.exists(sample_path):
            os.remove(sample_path)


def upload_file_multipart(url: str, file_path: str, fields: dict, timeout: int = 600) -> dict:
    """Tải file lên server bằng multipart/form-data qua urllib (không cần cài thêm thư viện)."""
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for k, v in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{v}\r\n".encode("utf-8"))

    filename = os.path.basename(file_path)
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: audio/wav\r\n\r\n")
    with open(file_path, "rb") as f:
        body.extend(f.read())
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("User-Agent", "ToolMovieAI-Client")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            err_msg = e.read().decode("utf-8", errors="ignore")[:300]
        except Exception:
            pass
        raise RuntimeError(f"Máy chủ Colab trả về HTTP {e.code}: {err_msg or e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Không thể kết nối đường truyền Cloudflare: {e.reason}")
    except OSError as e:
        raise RuntimeError(f"Đường truyền Colab bị gián đoạn ({e}). Vui lòng kiểm tra lại tab Google Colab!")


def process_with_colab(
    video_path: str,
    colab_url: str,
    model_size: str = "large-v3",
    separate_bgm: bool = True,
    output_srt: str = None,
    progress_cb = None
) -> tuple[str, str, str]:
    """
    Trích xuất âm thanh từ video cục bộ, gửi lên Colab GPU để tách BGM (Demucs v4)
    và nhận diện phụ đề (Faster-Whisper large-v3).
    Chỉ gửi file audio nhẹ (~10MB) nên cực nhanh và không tốn băng thông video.
    """
    base, _ = os.path.splitext(video_path)
    if not output_srt:
        output_srt = f"{base}_zh.srt"
    vocal_file = f"{base}_vocal.wav"
    instr_file = f"{base}_bgm.wav"

    colab_base = clean_colab_url(colab_url)
    if progress_cb:
        progress_cb(f"Đang kết nối tới Colab GPU Worker ({colab_base})...", 10)

    # Kiểm tra health trước khi trích xuất và tải file
    try:
        health = check_colab_worker(colab_base)
        gpu_name = health.get("gpu", {}).get("name", "GPU")
    except Exception as e:
        raise RuntimeError(f"Máy chủ Colab chưa phản hồi hoặc link Cloudflare đã hết hạn ({e}). Vui lòng kiểm tra lại tab Colab!")

    if progress_cb:
        progress_cb(f"[{gpu_name}] Đang nén âm thanh 16kHz siêu nhẹ...", 20)

    # 1. Trích xuất audio nén nhẹ (16kHz mono/stereo) để upload siêu tốc
    temp_raw = f"{base}_temp_upload.wav"
    subprocess.run([
        FFMPEG, "-y", "-i", video_path,
        "-vn", "-ac", "2", "-ar", "16000", "-c:a", "pcm_s16le", temp_raw
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    try:
        raw_mb = round(os.path.getsize(temp_raw) / (1024 * 1024), 1)
        if progress_cb:
            progress_cb(f"[{gpu_name}] Đang tải audio ({raw_mb}MB) lên Colab GPU...", 30)

        fields = {
            "model_size": model_size,
            "language": "zh",
            "separate_bgm": "true" if separate_bgm else "false",
        }

        # Dynamic progress ticker khi Colab đang xử lý Demucs v4 + Whisper
        stop_event = threading.Event()
        def ticker():
            stages = [
                (4, 40, f"[{gpu_name}] Colab đang tách bỏ giọng nói tiếng Trung (Demucs v4)..."),
                (14, 55, f"[{gpu_name}] Colab đang nhận diện phụ đề tiếng Trung (Whisper {model_size})..."),
                (28, 70, f"[{gpu_name}] Colab đang tinh chỉnh mốc thời gian phụ đề timestamp..."),
                (42, 78, f"[{gpu_name}] Colab đang hoàn tất xử lý kết quả..."),
            ]
            start_t = time.time()
            curr_idx = 0
            while not stop_event.is_set():
                elapsed = time.time() - start_t
                if curr_idx < len(stages):
                    delay, pct, msg = stages[curr_idx]
                    if elapsed >= delay:
                        if progress_cb:
                            progress_cb(msg, pct)
                        curr_idx += 1
                time.sleep(1)

        t_thread = threading.Thread(target=ticker, daemon=True)
        t_thread.start()

        try:
            res = upload_file_multipart(f"{colab_base}/api/process-all", temp_raw, fields)
        finally:
            stop_event.set()
            t_thread.join(timeout=1.0)

        srt_content = res.get("srt_content", "")
        with open(output_srt, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # 2. Tải file BGM tách sạch (Demucs v4) về máy Mac
        if res.get("has_bgm") and res.get("bgm_url"):
            bgm_dl_url = f"{colab_base}{res['bgm_url']}"
            if progress_cb:
                progress_cb(f"[{gpu_name}] Đang tải file BGM chất lượng cao (Demucs v4) về máy...", 85)
            req = urllib.request.Request(bgm_dl_url, headers={"User-Agent": "ToolMovieAI-Client"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(instr_file, "wb") as f_out:
                shutil.copyfileobj(resp, f_out)

        if res.get("vocal_url"):
            vocal_dl_url = f"{colab_base}{res['vocal_url']}"
            if progress_cb:
                progress_cb(f"[{gpu_name}] Đang lưu trữ file vocal tách riêng...", 95)
            req = urllib.request.Request(vocal_dl_url, headers={"User-Agent": "ToolMovieAI-Client"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(vocal_file, "wb") as f_out:
                shutil.copyfileobj(resp, f_out)

        if progress_cb:
            progress_cb("Trích xuất phụ đề & BGM qua Colab GPU hoàn tất!", 100)

        return srt_content, vocal_file, instr_file
    finally:
        if os.path.exists(temp_raw):
            os.remove(temp_raw)


def separate_audio_stems(video_path: str) -> tuple[str, str]:
    """
    Tách giọng nói (vocal) và nhạc nền (instrument/BGM) bằng mô hình ONNX Spleeter của pyvideotrans.
    Lưu cache cạnh file video để tái sử dụng tức thì ở bước lồng tiếng.
    """
    base, _ = os.path.splitext(video_path)
    vocal_file = f"{base}_vocal.wav"
    instr_file = f"{base}_bgm.wav"

    if os.path.exists(vocal_file) and os.path.exists(instr_file) and os.path.getsize(instr_file) > 1000:
        print(f"♻️ Đã có sẵn file tách nhạc nền & giọng nói trong cache.")
        return vocal_file, instr_file

    raw_wav = f"{base}_raw44k.wav"
    subprocess.run([
        FFMPEG, "-y", "-i", video_path,
        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", raw_wav
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    vocals_onnx = f"{PYVIDEOTRANS_DIR}/models/onnx/vocals.fp16.onnx"
    accomp_onnx = f"{PYVIDEOTRANS_DIR}/models/onnx/accompaniment.fp16.onnx"

    if os.path.exists(vocals_onnx) and os.path.exists(accomp_onnx):
        print("🎶 Đang tách giọng nói và nhạc nền bằng AI (Sherpa-ONNX Spleeter)...")
        import numpy as np
        import sherpa_onnx
        import soundfile as sf

        config = sherpa_onnx.OfflineSourceSeparationConfig(
            model=sherpa_onnx.OfflineSourceSeparationModelConfig(
                spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
                    vocals=vocals_onnx,
                    accompaniment=accomp_onnx,
                ),
                num_threads=4,
                debug=False,
                provider="cpu",
            )
        )
        sp = sherpa_onnx.OfflineSourceSeparation(config)
        samples, sample_rate = sf.read(raw_wav, dtype="float32", always_2d=True)
        samples = np.ascontiguousarray(np.transpose(samples))
        output = sp.process(sample_rate=sample_rate, samples=samples)

        vocals = np.transpose(output.stems[0].data)
        non_vocals = np.transpose(output.stems[1].data)
        sf.write(vocal_file, vocals, samplerate=output.sample_rate)
        sf.write(instr_file, non_vocals, samplerate=output.sample_rate)
        if os.path.exists(raw_wav):
            os.remove(raw_wav)
        print("✅ Tách nhạc nền hoàn tất!")
        return vocal_file, instr_file
    else:
        # Fallback nếu chưa có ONNX
        shutil.move(raw_wav, vocal_file)
        shutil.copy2(vocal_file, instr_file)
        return vocal_file, instr_file


def extract_subtitles(
    video_path: str,
    output_srt: str = None,
    model_size: str = "large-v3",
    separate_bgm: bool = True,
    colab_url: str = None,
    progress_cb = None
) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy file: {video_path}")

    if not output_srt:
        base, _ = os.path.splitext(video_path)
        output_srt = f"{base}_zh.srt"

    # Chỉ cho phép chạy trích xuất phụ đề trên Cloud GPU (Colab / Kaggle)
    colab_url = clean_colab_url(colab_url)
    if not colab_url:
        raise ValueError(
            "Chức năng trích xuất phụ đề chỉ cho phép chạy trên Cloud GPU (Google Colab / Kaggle). "
            "Vui lòng cung cấp Colab URL!"
        )

    target_model = model_size if model_size in ["large-v3", "medium", "small", "base"] else "large-v3"
    print(f"\n🚀 Đang gửi yêu cầu trích xuất phụ đề & tách BGM tới Colab GPU ({colab_url})...")
    srt_content, _, _ = process_with_colab(
        video_path=video_path,
        colab_url=colab_url,
        model_size=target_model,
        separate_bgm=separate_bgm,
        output_srt=output_srt,
        progress_cb=progress_cb
    )
    return srt_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất phụ đề tiếng Trung từ video ra file .srt")
    parser.add_argument("video", help="Đường dẫn file video (.mp4)")
    parser.add_argument("-o", "--output", help="Đường dẫn file .srt đầu ra (mặc định: cùng tên _zh.srt)")
    parser.add_argument("-m", "--model", default="large-v3", choices=["tiny", "base", "small", "medium", "large-v3"], help="Mô hình Whisper (mặc định: large-v3)")
    parser.add_argument("--no-separate", action="store_true", help="Không tách nhạc nền trước khi nhận diện")
    parser.add_argument("--colab-url", required=True, help="URL máy chủ Colab GPU Worker (ví dụ: https://xxx.trycloudflare.com)")
    args = parser.parse_args()

    extract_subtitles(args.video, args.output, args.model, separate_bgm=not args.no_separate, colab_url=args.colab_url)
