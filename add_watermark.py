#!/usr/bin/env python3
"""
Video Watermark & Personal Logo Overlay Tool
Chèn Logo hình ảnh (PNG trong suốt) hoặc Tên kênh cá nhân lên Video
"""

import os
import sys
import shutil
import argparse
import subprocess

POSITIONS = {
    "top-right": "W-w-15:15",
    "top-left": "15:15",
    "bottom-right": "W-w-15:H-h-15",
    "bottom-left": "15:H-h-15",
    "center": "(W-w)/2:(H-h)/2",
}

TEXT_POSITIONS = {
    "top-right": "w-tw-20:20",
    "top-left": "20:20",
    "bottom-right": "w-tw-20:h-th-20",
    "bottom-left": "20:h-th-20",
    "center": "(w-tw)/2:(h-th)/2",
}


def add_image_watermark(input_video: str, logo_image: str, output_video: str, pos: str = "top-right", width: int = 150, opacity: float = 1.0):
    ffmpeg_bin = shutil.which("ffmpeg") or "/Users/dungnguyenquoc/Dev/ToolAI/bin/ffmpeg"
    
    pos_expr = POSITIONS.get(pos, POSITIONS["top-right"])
    
    # Scale logo and adjust opacity
    if opacity < 1.0:
        filter_complex = f"[1:v]scale={width}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay={pos_expr}[v]"
    else:
        filter_complex = f"[1:v]scale={width}:-1[wm];[0:v][wm]overlay={pos_expr}[v]"
        
    cmd = [
        ffmpeg_bin, "-y",
        "-i", input_video,
        "-i", logo_image,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "h264_videotoolbox",
        "-c:a", "copy",
        output_video
    ]
    print(f"🎬 Đang chèn logo hình ảnh vào vị trí [{pos}]...")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        # Fallback to libx264 if videotoolbox fails
        cmd[cmd.index("h264_videotoolbox")] = "libx264"
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg chèn logo thất bại!")
    print(f"✅ Hoàn tất! File lưu tại: {output_video}")


def add_text_watermark(input_video: str, text: str, output_video: str, pos: str = "top-right", font_size: int = 24, font_color: str = "white"):
    ffmpeg_bin = shutil.which("ffmpeg") or "/Users/dungnguyenquoc/Dev/ToolAI/bin/ffmpeg"
    pos_expr = TEXT_POSITIONS.get(pos, TEXT_POSITIONS["top-right"])
    x_expr, y_expr = pos_expr.split(":")
    
    # Background box for readability
    vf = f"drawtext=text='{text}':fontsize={font_size}:fontcolor={font_color}:box=1:boxcolor=black@0.5:boxborderw=8:x={x_expr}:y={y_expr}"
    
    cmd = [
        ffmpeg_bin, "-y",
        "-i", input_video,
        "-vf", vf,
        "-c:v", "h264_videotoolbox",
        "-c:a", "copy",
        output_video
    ]
    print(f"🎬 Đang chèn chữ [{text}] vào vị trí [{pos}]...")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        cmd[cmd.index("h264_videotoolbox")] = "libx264"
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg chèn text thất bại!")
    print(f"✅ Hoàn tất! File lưu tại: {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Chèn Logo cá nhân hoặc Tên kênh lên Video")
    parser.add_argument("-i", "--input", required=True, help="Đường dẫn video đầu vào")
    parser.add_argument("-l", "--logo", help="Đường dẫn file ảnh logo (PNG trong suốt hoặc JPG)")
    parser.add_argument("-t", "--text", help="Chữ tên kênh / thương hiệu (nếu không có file ảnh logo)")
    parser.add_argument("-p", "--position", default="top-right", choices=["top-right", "top-left", "bottom-right", "bottom-left", "center"], help="Vị trí chèn (mặc định: top-right đè lên logo cũ)")
    parser.add_argument("-w", "--width", type=int, default=150, help="Độ rộng logo ảnh (px, mặc định: 150)")
    parser.add_argument("--opacity", type=float, default=0.9, help="Độ trong suốt logo (0.1 đến 1.0, mặc định: 0.9)")
    parser.add_argument("-o", "--output", help="Đường dẫn file video đầu ra (mặc định tự thêm _logo.mp4)")

    args = parser.parse_args()

    if not args.logo and not args.text:
        parser.error("Cần truyền ít nhất một trong hai tùy chọn: --logo (ảnh) hoặc --text (chữ)!")

    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_with_logo{ext}"

    if args.logo:
        add_image_watermark(args.input, args.logo, args.output, pos=args.position, width=args.width, opacity=args.opacity)
    else:
        add_text_watermark(args.input, args.text, args.output, pos=args.position)


if __name__ == "__main__":
    main()
