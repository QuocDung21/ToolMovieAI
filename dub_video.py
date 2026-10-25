#!/usr/bin/env python3
"""
Tool Lồng tiếng Video từ file Phụ đề Tiếng Việt (.srt)
Kế thừa core AI Vocal Separation (Spleeter ONNX), Edge-TTS và FFmpeg (libass + VideoToolbox).
"""

import os
import re
import sys
import shutil
import asyncio
import tempfile
import argparse
import subprocess
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


async def generate_speech_for_segment(
    text: str,
    voice: str,
    target_duration: float,
    output_wav: str,
    rate: str = "+0%",
    pitch: str = "+0Hz"
):
    """Generate TTS for single subtitle segment and time-stretch if needed."""
    import edge_tts

    temp_mp3 = output_wav + ".temp.mp3"
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await comm.save(temp_mp3)

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

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)


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

        inputs = ["-i", video_path]
        filter_parts = []
        current_v = "0:v"

        if hardsub:
            filter_parts.append(f"[{current_v}]subtitles=filename='{escaped_srt}'[vsub]")
            current_v = "vsub"

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
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg render thất bại: {res.stderr[-400:]}")

    report("Hoàn tất xuất video!", 100)
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
