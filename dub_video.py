#!/usr/bin/env python3
"""
Tool Lồng tiếng Video từ file Phụ đề Tiếng Việt (.srt)
Kế thừa core AI Vocal Separation (Spleeter ONNX), Edge-TTS và FFmpeg (libass + VideoToolbox).
"""

import os
import re
import sys
import json
import uuid
import shutil
import asyncio
import tempfile
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

from extract_sub import separate_audio_stems, FFMPEG

POSITIONS = {
    "top-right": "W-w-15:15",
    "top-left": "15:15",
    "bottom-right": "W-w-15:H-h-15",
    "bottom-left": "15:H-h-15",
    "center": "(W-w)/2:(H-h)/2",
}


def parse_srt(srt_file_or_content: str) -> list:
    """Parse SRT file path or raw SRT string into list of dicts: start, end, duration, text."""
    if os.path.exists(srt_file_or_content):
        with open(srt_file_or_content, "r", encoding="utf-8-sig") as f:
            content = f.read()
    else:
        content = srt_file_or_content

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


def get_media_duration(file_path: str) -> float:
    """Get duration of audio/video using ffmpeg."""
    cmd = [FFMPEG, "-i", file_path]
    res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', res.stderr)
    if m:
        h, mn, s = m.groups()
        return int(h) * 3600 + int(mn) * 60 + float(s)
    return 0.0


def sanitize_tts_text(text: str) -> str:
    """Clean subtitle text and remove HTML/markers for TTS synthesis."""
    if not text:
        return ""
    cleaned = re.sub(r'<[^>]+>', '', text)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def generate_silence_wav(duration: float, output_wav: str):
    """Generate silent wav of exact duration as a fallback."""
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
    """Generate TTS for single subtitle segment with retry, text variant recovery, and time-stretch."""
    import edge_tts

    t_clean = sanitize_tts_text(text)
    if not t_clean or not any(c.isalnum() for c in t_clean):
        # Empty or purely non-alphanumeric text
        generate_silence_wav(target_duration, output_wav)
        return

    # Build candidates: Edge-TTS Vietnamese parser sometimes requires or rejects specific trailing punctuation
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
        print(f"⚠️ [TTS Warning] Không nhận được audio cho câu: {text[:60]!r} ({last_error}). Đang dùng khoảng lặng.")
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


async def run_dubbing_pipeline(
    video_path: str,
    srt_path: str,
    output_path: str,
    voice: str = "vi-VN-NamMinhNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    bgm_volume: float = 0.30,
    separate_bgm: bool = True,
    hardsub: bool = True,
    logo_image: str = None,
    logo_pos: str = "top-right",
    logo_width: int = 150,
    progress_cb=None
) -> str:
    def report(msg: str, pct: int):
        print(f"[{pct}%] {msg}")
        if progress_cb:
            progress_cb(msg, pct)

    total_duration = get_media_duration(video_path)
    subs = parse_srt(srt_path)
    if not subs:
        raise ValueError("File phụ đề không hợp lệ hoặc không có câu thoại nào!")

    report(f"Bắt đầu tạo giọng đọc AI ({voice}) cho {len(subs)} câu thoại...", 10)

    with tempfile.TemporaryDirectory() as temp_dir:
        seg_files = []
        for idx, sub in enumerate(subs, start=1):
            seg_wav = os.path.join(temp_dir, f"seg_{idx:04d}.wav")
            pct = 10 + int((idx / len(subs)) * 45)
            report(f"Đang đọc câu {idx}/{len(subs)}: {sub['text'][:45]}...", pct)
            await generate_speech_for_segment(
                sub["text"], voice, sub["duration"], seg_wav, rate=rate, pitch=pitch
            )
            seg_files.append((sub["start"], seg_wav))
            await asyncio.sleep(0.25)

        report("Đang căn chỉnh các câu thoại vào đúng mốc thời gian...", 60)
        voice_track = os.path.join(temp_dir, "voice_track.wav")
        
        # Đặt các câu thoại lên timeline bằng numpy/soundfile để tránh giới hạn số input của FFmpeg khi video dài (hàng trăm câu)
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

        # Chống vỡ tiếng (clipping)
        timeline = np.clip(timeline, -0.98, 0.98)
        sf.write(voice_track, timeline[:int(total_duration * sr)], sr)

        # Xử lý nhạc nền BGM
        bgm_track = os.path.join(temp_dir, "bgm_scaled.wav")
        if separate_bgm:
            report("Đang tách bỏ tiếng Trung gốc, giữ lại nhạc nền & hiệu ứng (AI Spleeter)...", 70)
            try:
                _, clean_instr = separate_audio_stems(video_path)
                source_for_bgm = clean_instr
            except Exception as e:
                print(f"⚠️ Lỗi tách BGM ({e}), dùng audio gốc hạ âm lượng.")
                source_for_bgm = video_path
        else:
            report("Đang hạ âm lượng âm thanh gốc làm nền...", 70)
            source_for_bgm = video_path

        subprocess.run([
            FFMPEG, "-y", "-i", source_for_bgm,
            "-vn", "-filter:a", f"volume={bgm_volume:.2f}",
            "-ar", "44100", "-ac", "2", bgm_track
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Trộn giọng đọc + nhạc nền
        report("Đang hòa âm giọng lồng tiếng Việt và nhạc nền...", 80)
        mixed_audio = os.path.join(temp_dir, "mixed_audio.wav")
        subprocess.run([
            FFMPEG, "-y",
            "-i", voice_track,
            "-i", bgm_track,
            "-filter_complex", "[0:a]volume=1.15[v];[1:a]volume=1.0[bg];[v][bg]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:level=0[aout]",
            "-map", "[aout]",
            mixed_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Render video hoàn chỉnh (Hardsub + Logo)
        report("Đang xuất video hoàn chỉnh (in phụ đề & gắn logo)...", 90)
        pos_expr = POSITIONS.get(logo_pos, POSITIONS["top-right"])

        # Copy srt vào temp_dir với tên đơn giản để tránh lỗi ký tự đặc biệt trong filter subtitles của FFmpeg
        safe_srt = os.path.join(temp_dir, "sub.srt")
        shutil.copy2(srt_path, safe_srt)
        # Escape path for ffmpeg filter
        escaped_srt = safe_srt.replace("\\", "/").replace(":", "\\:")

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
                filter_parts.append(f"[{current_v}]subtitles=filename='{escaped_srt}'[vsub]")
                current_v = "vsub"
            else:
                print("⚠️ FFmpeg không hỗ trợ filter 'subtitles' (libass), bỏ qua hardsub và tiếp tục xuất video.")

        if logo_image and os.path.exists(logo_image):
            inputs.extend(["-i", logo_image])
            logo_idx = 1
            filter_parts.append(f"[{logo_idx}:v]scale={logo_width}:-1[wm]")
            filter_parts.append(f"[{current_v}][wm]overlay={pos_expr}[vout]")
            current_v = "vout"

        audio_input_idx = len(inputs) // 2
        inputs.extend(["-i", mixed_audio])

        cmd = [FFMPEG, "-y"] + inputs
        if filter_parts:
            cmd.extend([
                "-filter_complex", ";".join(filter_parts),
                "-map", f"[{current_v}]",
                "-map", f"{audio_input_idx}:a",
                "-c:v", "h264_videotoolbox",
                "-b:v", "2500k",
            ])
        else:
            cmd.extend([
                "-map", "0:v",
                "-map", f"{audio_input_idx}:a",
                "-c:v", "copy",
            ])

        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest", output_path])

        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 and "h264_videotoolbox" in cmd:
            cmd[cmd.index("h264_videotoolbox")] = "libx264"
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        # Fallback an toàn nếu filter phức tạp thất bại: xuất video cơ bản với âm thanh đã lồng tiếng
        if res.returncode != 0:
            print(f"⚠️ Render filter thất bại ({res.stderr[-200:] if res.stderr else ''}), chuyển sang chế độ fallback không filter...")
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
                output_path
            ]
            fb_res = subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if fb_res.returncode != 0:
                raise RuntimeError(f"FFmpeg render thất bại hoàn toàn: {fb_res.stderr[-300:]}")

    report("Hoàn tất xuất video!", 100)
    return output_path


def upload_dubbing_to_colab(
    url: str,
    video_path: str,
    fields: dict,
    logo_path: str = None,
    timeout: int = 1800
) -> dict:
    """Tải video + phụ đề lên server Colab GPU để thực hiện lồng tiếng & render video hoàn chỉnh."""
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

    with tempfile.TemporaryFile() as tmp:
        for k, v in fields.items():
            tmp.write(f"--{boundary}\r\n".encode("utf-8"))
            tmp.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
            tmp.write(f"{v}\r\n".encode("utf-8"))

        v_name = os.path.basename(video_path)
        tmp.write(f"--{boundary}\r\n".encode("utf-8"))
        tmp.write(f'Content-Disposition: form-data; name="file"; filename="{v_name}"\r\n'.encode("utf-8"))
        tmp.write(b"Content-Type: video/mp4\r\n\r\n")
        with open(video_path, "rb") as vf:
            shutil.copyfileobj(vf, tmp)
        tmp.write(b"\r\n")

        if logo_path and os.path.exists(logo_path):
            l_name = os.path.basename(logo_path)
            tmp.write(f"--{boundary}\r\n".encode("utf-8"))
            tmp.write(f'Content-Disposition: form-data; name="logo_file"; filename="{l_name}"\r\n'.encode("utf-8"))
            tmp.write(b"Content-Type: image/png\r\n\r\n")
            with open(logo_path, "rb") as lf:
                shutil.copyfileobj(lf, tmp)
            tmp.write(b"\r\n")

        tmp.write(f"--{boundary}--\r\n".encode("utf-8"))

        tmp.seek(0, os.SEEK_END)
        content_len = tmp.tell()
        tmp.seek(0)

        req = urllib.request.Request(url, data=tmp)
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Content-Length", str(content_len))
        req.add_header("User-Agent", "ToolMovieAI-Client")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_msg = ""
            try:
                err_msg = e.read().decode("utf-8", errors="ignore")[:400]
            except Exception:
                pass
            raise RuntimeError(f"Máy chủ Colab báo lỗi HTTP {e.code}: {err_msg or e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Không thể kết nối đường truyền Cloudflare: {e.reason}")
        except OSError as e:
            raise RuntimeError(f"Đường truyền Colab bị gián đoạn ({e}). Vui lòng kiểm tra lại tab Colab!")


def process_dubbing_with_colab(
    video_path: str,
    srt_content: str,
    output_path: str,
    colab_url: str,
    voice: str = "vi-VN-NamMinhNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
    bgm_volume: float = 0.30,
    separate_bgm: bool = True,
    hardsub: bool = True,
    logo_image: str = None,
    logo_pos: str = "top-right",
    logo_width: int = 150,
    progress_cb=None
) -> str:
    """Ủy thác toàn bộ pipeline lồng tiếng và render video lên GPU Google Colab / Kaggle."""
    colab_base = colab_url.rstrip("/")
    if progress_cb:
        progress_cb(f"Đang kết nối tới Colab GPU Worker ({colab_base})...", 10)

    from extract_sub import check_colab_worker
    check_colab_worker(colab_base)

    if progress_cb:
        v_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 1)
        progress_cb(f"Đang tải video ({v_size_mb}MB) & phụ đề lên Colab GPU...", 20)

    fields = {
        "srt_content": srt_content,
        "voice": voice,
        "rate": rate,
        "pitch": pitch,
        "bgm_volume": str(bgm_volume),
        "separate_bgm": "true" if separate_bgm else "false",
        "hardsub": "true" if hardsub else "false",
        "logo_pos": logo_pos,
        "logo_width": str(logo_width),
    }

    if progress_cb:
        progress_cb("Colab GPU đang xử lý tạo giọng Edge-TTS, hòa âm BGM và render video...", 45)

    res = upload_dubbing_to_colab(
        url=f"{colab_base}/api/dub-video",
        video_path=video_path,
        fields=fields,
        logo_path=logo_image,
        timeout=1800
    )

    if not res.get("download_url"):
        raise RuntimeError(f"Colab không trả về link tải video đã lồng tiếng: {res}")

    if progress_cb:
        progress_cb("Colab đã render hoàn tất! Đang tải video thành phẩm về máy...", 85)

    dl_url = f"{colab_base}{res['download_url']}"
    req = urllib.request.Request(dl_url, headers={"User-Agent": "ToolMovieAI-Client"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(output_path, "wb") as f_out:
        shutil.copyfileobj(resp, f_out)

    if progress_cb:
        progress_cb("Hoàn tất xuất video qua Colab GPU!", 100)

    return output_path


async def main_async():
    parser = argparse.ArgumentParser(
        description="Lồng tiếng Video từ file Phụ đề Tiếng Việt (.srt)"
    )
    parser.add_argument("video", help="Đường dẫn file video (.mp4)")
    parser.add_argument("subtitles", help="Đường dẫn file phụ đề tiếng Việt (.srt)")
    parser.add_argument("-v", "--voice", default="vi-VN-NamMinhNeural", choices=["vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"], help="Giọng lồng tiếng")
    parser.add_argument("--rate", default="+0%", help="Tốc độ giọng đọc (ví dụ: +0%%, -5%%, +10%%)")
    parser.add_argument("--pitch", default="+0Hz", help="Cao độ giọng đọc (ví dụ: +0Hz, -2Hz, +3Hz)")
    parser.add_argument("-l", "--logo", help="Đường dẫn file logo PNG chèn góc video")
    parser.add_argument("-p", "--logo-pos", default="top-right", choices=list(POSITIONS.keys()), help="Vị trí logo")
    parser.add_argument("-b", "--bgm-volume", type=float, default=0.30, help="Âm lượng nhạc nền gốc (mặc định: 0.30)")
    parser.add_argument("--no-separate", action="store_true", help="Không tách giọng Trung bằng AI mà chỉ hạ nhỏ âm gốc")
    parser.add_argument("--no-hardsub", action="store_true", help="Không in phụ đề cứng lên video")
    parser.add_argument("-o", "--output", help="Đường dẫn file video hoàn chỉnh đầu ra")

    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.video)
        args.output = f"{base}_dubbed{ext}"

    await run_dubbing_pipeline(
        video_path=args.video,
        srt_path=args.subtitles,
        output_path=args.output,
        voice=args.voice,
        rate=args.rate,
        pitch=args.pitch,
        bgm_volume=args.bgm_volume,
        separate_bgm=not args.no_separate,
        hardsub=not args.no_hardsub,
        logo_image=args.logo,
        logo_pos=args.logo_pos,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        sys.exit(1)
