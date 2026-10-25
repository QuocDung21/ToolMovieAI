#!/usr/bin/env python3
"""
Tool trích xuất phụ đề tiếng Trung từ Video (Speech-to-Text) & Tách Nhạc Nền
Kế thừa trực tiếp mô hình Faster-Whisper và Sherpa-ONNX Spleeter từ pyvideotrans core.
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

PYVIDEOTRANS_DIR = "/Users/dungnguyenquoc/Dev/ToolAI/pyvideotrans"
FFMPEG = f"{PYVIDEOTRANS_DIR}/ffmpeg/ffmpeg"
if not os.path.exists(FFMPEG):
    FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


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


def extract_subtitles(video_path: str, output_srt: str = None, model_size: str = "base", separate_bgm: bool = True) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy file: {video_path}")

    if not output_srt:
        base, _ = os.path.splitext(video_path)
        output_srt = f"{base}_zh.srt"

    audio_input = video_path
    if separate_bgm:
        try:
            vocal_wav, _ = separate_audio_stems(video_path)
            audio_input = vocal_wav
        except Exception as e:
            print(f"⚠️ Bỏ qua bước tách vocal riêng ({e}), dùng trực tiếp audio gốc.")

    print(f"\n🎙️ Đang nhận diện lời thoại tiếng Trung bằng Faster-Whisper ({model_size})...")
    from faster_whisper import WhisperModel

    # Ưu tiên lấy model đã tải sẵn trong pyvideotrans/models
    local_model_dir = f"{PYVIDEOTRANS_DIR}/models/models--Systran--faster-whisper-{model_size}"
    snapshots_dir = os.path.join(local_model_dir, "snapshots")
    model_path_or_name = model_size
    if os.path.exists(snapshots_dir):
        snaps = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir) if not d.startswith(".")]
        if snaps:
            model_path_or_name = snaps[0]

    model = WhisperModel(model_path_or_name, device="cpu", compute_type="float32")

    segments, info = model.transcribe(
        audio_input,
        language="zh",
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=False,
    )

    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = seg.start
        end = seg.end
        text = seg.text.strip()

        start_h = int(start // 3600)
        start_m = int((start % 3600) // 60)
        start_s = int(start % 60)
        start_ms = int(round((start - int(start)) * 1000))

        end_h = int(end // 3600)
        end_m = int((end % 3600) // 60)
        end_s = int(end % 60)
        end_ms = int(round((end - int(end)) * 1000))

        time_line = f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> {end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}"
        lines.append(f"{idx}\n{time_line}\n{text}\n")
        print(f"[{time_line}] {text}")

    srt_content = "\n".join(lines)
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"\n✅ Xuất file phụ đề tiếng Trung thành công: {os.path.abspath(output_srt)}")
    return srt_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất phụ đề tiếng Trung từ video ra file .srt")
    parser.add_argument("video", help="Đường dẫn file video (.mp4)")
    parser.add_argument("-o", "--output", help="Đường dẫn file .srt đầu ra (mặc định: cùng tên _zh.srt)")
    parser.add_argument("-m", "--model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"], help="Mô hình Whisper (mặc định: base)")
    parser.add_argument("--no-separate", action="store_true", help="Không tách nhạc nền trước khi nhận diện")
    args = parser.parse_args()

    extract_subtitles(args.video, args.output, args.model, separate_bgm=not args.no_separate)
