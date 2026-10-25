#!/usr/bin/env python3
"""
Bilibili Video & Subtitle Downloader
Custom-built downloader bypassing Bilibili 412 / 403 / anti-bot restrictions.
Uses optimized headers and native curl/ffmpeg streaming.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

# Headers for API requests (Bilibili blocks Referer on unauthenticated API calls)
API_HEADERS = {
    "User-Agent": "curl/8.7.1",
    "Accept": "*/*",
}

QUALITY_MAP = {
    127: "8K 超高清",
    120: "4K 超清",
    116: "1080P 60帧",
    112: "1080P 高码率",
    80: "1080P 高清",
    64: "720P 高清",
    32: "480P 清晰",
    16: "360P 流畅",
}


def sanitize_filename(name: str) -> str:
    """Loại bỏ ký tự không hợp lệ trong tên file."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def parse_bvid(url_or_id: str) -> str:
    """Trích xuất BV hoặc AV ID từ URL hoặc chuỗi nhập vào."""
    url_or_id = url_or_id.strip()
    match = re.search(r"(BV[a-zA-Z0-9]{10})", url_or_id, re.IGNORECASE)
    if match:
        return match.group(1)

    match_av = re.search(r"av(\d+)", url_or_id, re.IGNORECASE)
    if match_av:
        return f"av{match_av.group(1)}"

    raise ValueError(f"Không nhận diện được BV hoặc AV ID từ: {url_or_id}")


def make_api_request(url: str, cookie: str = "") -> dict:
    """Gửi request lấy dữ liệu JSON từ Bilibili API với header curl."""
    headers = dict(API_HEADERS)
    if cookie:
        headers["Cookie"] = cookie

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {e.reason} - {body[:200]}")
    except Exception as e:
        raise RuntimeError(f"Lỗi kết nối tới Bilibili: {e}")


def download_with_curl(url: str, output_path: str, desc: str = "Tải xuống"):
    """Tải file stream từ Bilibili CDN bằng curl với Referer chuẩn."""
    print(f"\n📥 {desc}...")
    curl_bin = shutil.which("curl") or "/usr/bin/curl"

    cmd = [
        curl_bin,
        "-L",
        "--progress-bar",
        "-H",
        "Referer: https://www.bilibili.com/",
        "-H",
        "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "-o",
        output_path,
        url,
    ]

    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Tải file thất bại (curl exit code: {res.returncode})")


def json_sub_to_srt(json_sub: dict) -> str:
    """Chuyển đổi phụ đề Bilibili CC (JSON) sang định dạng SRT chuẩn."""
    lines = []
    body = json_sub.get("body", [])
    for idx, item in enumerate(body, start=1):
        start = float(item.get("from", 0))
        end = float(item.get("to", 0))
        text = item.get("content", "").strip()

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
    return "\n".join(lines)


def get_video_info(bvid: str, cookie: str = "") -> dict:
    """Lấy thông tin video và danh sách các tập (pages)."""
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    resp = make_api_request(url, cookie=cookie)
    if resp.get("code") != 0:
        raise RuntimeError(
            f"Bilibili API trả về lỗi: {resp.get('message')} (mã {resp.get('code')})"
        )
    return resp["data"]


def get_subtitles(bvid: str, cid: int, cookie: str = "") -> list:
    """Lấy danh sách phụ đề CC có sẵn."""
    url = f"https://api.bilibili.com/x/player/v2?bvid={bvid}&cid={cid}"
    resp = make_api_request(url, cookie=cookie)
    if resp.get("code") == 0 and "data" in resp:
        sub_info = resp["data"].get("subtitle", {})
        return sub_info.get("subtitles", [])
    return []


def get_playurl(
    bvid: str, cid: int, qn: int = 80, prefer_dash: bool = True, cookie: str = ""
) -> dict:
    """Lấy link luồng phát video (hỗ trợ cả DASH và DURL)."""
    fnval = 16 if prefer_dash else 0  # 16 = DASH, 0 = DURL
    url = f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}&qn={qn}&fnval={fnval}&fnver=0&fourk=1"
    resp = make_api_request(url, cookie=cookie)
    if resp.get("code") != 0:
        # Fallback về DURL nếu DASH không khả dụng
        url = f"https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={cid}&qn={qn}&fnval=0"
        resp = make_api_request(url, cookie=cookie)
        if resp.get("code") != 0:
            raise RuntimeError(
                f"Lỗi lấy link phát video: {resp.get('message')} (mã {resp.get('code')})"
            )
    return resp["data"]


def main():
    parser = argparse.ArgumentParser(
        description="Bilibili Video & Subtitle Downloader - Tool tải Bilibili tốc độ cao, không bị chặn 412"
    )
    parser.add_argument(
        "url", help="URL video Bilibili hoặc mã BV (ví dụ: BV1xx411c7mD)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./downloads",
        help="Thư mục lưu video (mặc định: ./downloads)",
    )
    parser.add_argument(
        "-p",
        "--page",
        type=int,
        default=1,
        help="Số thứ tự phân đoạn / tập (mặc định: 1)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=80,
        help="Chất lượng ưu tiên (120: 4K, 80: 1080P, 64: 720P, 32: 480P)",
    )
    parser.add_argument(
        "-c",
        "--cookie",
        default="",
        help="Chuỗi Cookie (dùng để tải 1080P+ hoặc tài khoản VIP)",
    )
    parser.add_argument(
        "--sub-only", action="store_true", help="Chỉ tải phụ đề CC (.srt) nếu có"
    )
    parser.add_argument(
        "--info", action="store_true", help="Chỉ xem thông tin video mà không tải"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bvid = parse_bvid(args.url)
    print(f"\n[1/4] Đang lấy thông tin video: {bvid}...")
    vinfo = get_video_info(bvid, cookie=args.cookie)

    title = vinfo.get("title", bvid)
    author = vinfo.get("owner", {}).get("name", "Unknown")
    pages = vinfo.get("pages", [])

    print(f"🎬 Tiêu đề: {title}")
    print(f"👤 Người đăng: {author}")
    print(f"📑 Số tập/phần: {len(pages)}")

    target_page = None
    for p in pages:
        if p.get("page") == args.page:
            target_page = p
            break
    if not target_page:
        target_page = pages[0]

    cid = target_page["cid"]
    part_title = target_page.get("part", "")
    full_title = (
        f"{title}_{part_title}" if part_title and part_title != title else title
    )
    safe_title = sanitize_filename(full_title)

    print(
        f"🎯 Phân đoạn chọn: P{target_page.get('page', 1)} - {part_title} (CID: {cid})"
    )

    # Kiểm tra và tải phụ đề CC nếu có
    print("\n[2/4] Đang kiểm tra phụ đề CC gốc...")
    subs = get_subtitles(bvid, cid, cookie=args.cookie)
    if subs:
        print(f"✅ Tìm thấy {len(subs)} phụ đề CC:")
        for s in subs:
            lang = s.get("lan_doc", s.get("lan", ""))
            sub_url = s.get("subtitle_url", "")
            if sub_url.startswith("//"):
                sub_url = "https:" + sub_url
            print(f"   • {lang}: {sub_url}")

            try:
                sub_data = make_api_request(sub_url)
                srt_content = json_sub_to_srt(sub_data)
                srt_filename = os.path.join(args.output_dir, f"{safe_title}_{lang}.srt")
                with open(srt_filename, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                print(f"   💾 Đã lưu phụ đề SRT: {srt_filename}")
            except Exception as e:
                print(f"   ⚠️ Lỗi tải phụ đề {lang}: {e}")
    else:
        print(
            "ℹ️ Video không có phụ đề CC gốc (sẽ dùng ASR Whisper để nhận diện giọng nói khi lồng tiếng)."
        )

    if args.sub_only or args.info:
        print("\n✨ Hoàn tất theo yêu cầu.")
        return

    # Lấy thông tin luồng video/audio
    print("\n[3/4] Đang lấy luồng video/audio...")
    play_data = get_playurl(
        bvid, cid, qn=args.quality, prefer_dash=True, cookie=args.cookie
    )

    accept_desc = play_data.get("accept_description", [])
    accept_q = play_data.get("accept_quality", [])
    print(f"📺 Các mức chất lượng khả dụng: {', '.join(accept_desc)} (Mã: {accept_q})")

    final_output = os.path.join(args.output_dir, f"{safe_title}.mp4")

    # Xử lý luồng DASH (chất lượng cao nhất: video + audio tách riêng)
    if "dash" in play_data:
        dash = play_data["dash"]
        video_streams = sorted(
            dash.get("video", []), key=lambda x: x.get("id", 0), reverse=True
        )
        audio_streams = sorted(
            dash.get("audio", []), key=lambda x: x.get("id", 0), reverse=True
        )

        if not video_streams:
            raise RuntimeError("Không tìm thấy luồng video trong DASH.")

        sel_video = video_streams[0]
        sel_audio = audio_streams[0] if audio_streams else None

        v_url = sel_video.get("baseUrl") or sel_video.get("base_url")
        a_url = (
            sel_audio.get("baseUrl") or sel_audio.get("base_url") if sel_audio else None
        )

        q_name = QUALITY_MAP.get(sel_video.get("id"), f"Q{sel_video.get('id')}")
        print(f"🎥 Đã chọn chất lượng: {q_name}")

        tmp_v = os.path.join(args.output_dir, f"_temp_{bvid}_video.m4s")
        tmp_a = os.path.join(args.output_dir, f"_temp_{bvid}_audio.m4s")

        print("\n[4/4] Bắt đầu tải các luồng media...")
        download_with_curl(v_url, tmp_v, desc=f"Tải luồng Hình ảnh ({q_name})")
        if a_url:
            download_with_curl(a_url, tmp_a, desc="Tải luồng Âm thanh")

        print("\n⚡ Đang ghép hình ảnh và âm thanh qua FFmpeg...")
        ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if not ffmpeg_bin or not os.path.exists(ffmpeg_bin):
            raise RuntimeError("Không tìm thấy FFmpeg trên máy!")

        cmd = [ffmpeg_bin, "-y", "-i", tmp_v]
        if a_url:
            cmd.extend(["-i", tmp_a, "-c:v", "copy", "-c:a", "copy"])
        else:
            cmd.extend(["-c:v", "copy"])
        cmd.append(final_output)

        res = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )

        # Dọn dẹp file tạm
        if os.path.exists(tmp_v):
            os.remove(tmp_v)
        if os.path.exists(tmp_a):
            os.remove(tmp_a)

        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg ghép video thất bại: {res.stderr}")

    # Xử lý luồng DURL (File MP4 đã trộn sẵn)
    elif "durl" in play_data:
        durls = play_data["durl"]
        v_url = durls[0]["url"]
        q_name = QUALITY_MAP.get(play_data.get("quality"), "Chuẩn")
        print(f"🎥 Định dạng: MP4 nguyên bản ({q_name})")
        print("\n[4/4] Bắt đầu tải video...")
        download_with_curl(v_url, final_output, desc=f"Tải Video MP4 ({q_name})")

    print(f"\n🎉 Tải video thành công!")
    print(f"📁 File hoàn chỉnh: {os.path.abspath(final_output)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        sys.exit(1)
