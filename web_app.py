#!/usr/bin/env python3
"""
Bilibili AI Sub & Dubbing Studio - Web Application
Kế thừa core từ pyvideotrans (Faster-Whisper, Sherpa-ONNX Vocal Separation, Edge-TTS, FFmpeg)
và bili_downloader.py.
"""

import os
import re
import uuid
import json
import time
import shutil
import subprocess
import urllib.parse
import urllib.request
import asyncio
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bili_downloader import (
    parse_bvid,
    get_video_info,
    get_subtitles,
    get_playurl,
    download_with_curl,
    json_sub_to_srt,
    make_api_request,
    sanitize_filename,
    QUALITY_MAP,
)
from extract_sub import extract_subtitles, separate_audio_stems, check_colab_worker, FFMPEG
from dub_video import run_dubbing_pipeline, parse_srt

BASE_DIR = "/Users/dungnguyenquoc/Dev/ToolAI"
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGOS_DIR = os.path.join(BASE_DIR, "logos")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGOS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Copy default sample_logo.png into logos dir if present
default_logo = os.path.join(BASE_DIR, "sample_logo.png")
if os.path.exists(default_logo) and not os.path.exists(os.path.join(LOGOS_DIR, "sample_logo.png")):
    shutil.copy2(default_logo, os.path.join(LOGOS_DIR, "sample_logo.png"))

app = FastAPI(title="Bilibili AI Sub & Dubbing Studio")

# Mount static media directories for video/audio preview in browser
app.mount("/downloads", StaticFiles(directory=DOWNLOADS_DIR), name="downloads")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/logos", StaticFiles(directory=LOGOS_DIR), name="logos")

# In-memory background task tracker
TASKS = {}


def create_task(title: str) -> str:
    tid = str(uuid.uuid4())[:8]
    TASKS[tid] = {
        "id": tid,
        "title": title,
        "status": "running",  # running | done | error
        "progress": 5,
        "message": "Đang khởi tạo...",
        "result": None,
        "error": None,
    }
    return tid


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------
class BiliInfoReq(BaseModel):
    url: str
    cookie: Optional[str] = ""


class BiliDownloadReq(BaseModel):
    url: str
    page: int = 1
    quality: int = 80
    cookie: Optional[str] = ""


class ExtractSubReq(BaseModel):
    video_filename: str
    model_size: str = "base"
    separate_bgm: bool = True
    colab_url: Optional[str] = None


class ColabCheckReq(BaseModel):
    colab_url: str


class AutoTranslateReq(BaseModel):
    srt_content: str
    provider: str = "google"  # google | llm
    api_url: Optional[str] = "https://api.deepseek.com/v1"
    api_key: Optional[str] = ""
    model_name: Optional[str] = "deepseek-chat"
    custom_prompt: Optional[str] = ""


class PreviewVoiceReq(BaseModel):
    text: str
    voice: str = "vi-VN-NamMinhNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


class TextLogoReq(BaseModel):
    text: str
    bg_color: str = "#141419"
    border_color: str = "#00c8ff"


class DubRenderReq(BaseModel):
    video_filename: str
    vi_srt_content: str
    voice: str = "vi-VN-NamMinhNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    bgm_volume: float = 0.25
    separate_bgm: bool = True
    hardsub: bool = True
    burn_hardsub: Optional[bool] = None
    logo_filename: Optional[str] = ""
    logo_pos: str = "top-right"
    logo_position: Optional[str] = None
    logo_width: int = 150


class SaveSubtitleFileReq(BaseModel):
    filename: str
    content: str


class SaveSubtitlePairReq(BaseModel):
    video_filename: str
    zh_srt: Optional[str] = None
    vi_srt: Optional[str] = None


class LibraryRenameReq(BaseModel):
    category: str  # download | output | subtitle
    old_filename: str
    new_filename: str


class LibraryDeleteReq(BaseModel):
    category: str  # download | output | subtitle
    filename: str
    delete_related: bool = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h3>Đang khởi tạo giao diện...</h3>")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Không tìm thấy tiến trình")
    return TASKS[task_id]


@app.get("/api/local-files")
async def list_local_files():
    videos = []
    for f in sorted(os.listdir(DOWNLOADS_DIR)):
        if f.lower().endswith(".mp4") and not f.startswith("_temp_"):
            full_p = os.path.join(DOWNLOADS_DIR, f)
            base, _ = os.path.splitext(full_p)
            zh_srt = f"{base}_zh.srt"
            vi_srt = f"{base}_vi.srt"
            bgm_wav = f"{base}_bgm.wav"
            size_mb = round(os.path.getsize(full_p) / (1024 * 1024), 1)
            videos.append({
                "filename": f,
                "url": f"/downloads/{urllib.parse.quote(f)}",
                "size_mb": size_mb,
                "has_zh_srt": os.path.exists(zh_srt),
                "has_vi_srt": os.path.exists(vi_srt),
                "has_bgm": os.path.exists(bgm_wav),
            })

    logos = [f for f in sorted(os.listdir(LOGOS_DIR)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    outputs = []
    for f in sorted(os.listdir(OUTPUTS_DIR), reverse=True):
        if f.lower().endswith(".mp4"):
            full_p = os.path.join(OUTPUTS_DIR, f)
            outputs.append({
                "filename": f,
                "url": f"/outputs/{urllib.parse.quote(f)}",
                "size_mb": round(os.path.getsize(full_p) / (1024 * 1024), 1),
            })

    subtitles = []
    for f in sorted(os.listdir(DOWNLOADS_DIR)):
        if f.lower().endswith(".srt"):
            full_p = os.path.join(DOWNLOADS_DIR, f)
            try:
                content = Path(full_p).read_text(encoding="utf-8", errors="ignore")
                line_count = len(re.findall(r"-->", content))
            except Exception:
                line_count = 0
            lang = "vi" if f.lower().endswith("_vi.srt") else ("zh" if f.lower().endswith("_zh.srt") else "other")
            subtitles.append({
                "filename": f,
                "url": f"/downloads/{urllib.parse.quote(f)}",
                "size_kb": round(os.path.getsize(full_p) / 1024, 1),
                "line_count": line_count,
                "lang": lang,
            })

    return {"videos": videos, "logos": logos, "outputs": outputs, "subtitles": subtitles}


@app.get("/api/subtitles/{video_filename}")
async def get_saved_subtitles(video_filename: str):
    video_path = os.path.join(DOWNLOADS_DIR, video_filename)
    base, _ = os.path.splitext(video_path)
    zh_path = f"{base}_zh.srt"
    vi_path = f"{base}_vi.srt"

    zh_content = Path(zh_path).read_text(encoding="utf-8") if os.path.exists(zh_path) else ""
    vi_content = Path(vi_path).read_text(encoding="utf-8") if os.path.exists(vi_path) else ""
    return {"zh_srt": zh_content, "vi_srt": vi_content}


@app.post("/api/subtitles/save-pair")
async def save_subtitle_pair(req: SaveSubtitlePairReq):
    base_name = os.path.basename(req.video_filename)
    video_path = os.path.join(DOWNLOADS_DIR, base_name)
    base, _ = os.path.splitext(video_path)
    if req.zh_srt is not None:
        Path(f"{base}_zh.srt").write_text(req.zh_srt.strip(), encoding="utf-8")
    if req.vi_srt is not None:
        Path(f"{base}_vi.srt").write_text(req.vi_srt.strip(), encoding="utf-8")
    return {"status": "ok"}


@app.get("/api/library/subtitle/{filename}")
async def get_single_subtitle(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(DOWNLOADS_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file phụ đề")
    content = Path(path).read_text(encoding="utf-8", errors="ignore")
    return {"filename": safe_name, "content": content}


@app.post("/api/library/subtitle/save")
async def save_single_subtitle(req: SaveSubtitleFileReq):
    safe_name = os.path.basename(req.filename.strip())
    if not safe_name.lower().endswith(".srt"):
        safe_name += ".srt"
    path = os.path.join(DOWNLOADS_DIR, safe_name)
    Path(path).write_text(req.content.strip(), encoding="utf-8")
    return {"status": "ok", "filename": safe_name}


@app.post("/api/library/rename")
async def rename_library_file(req: LibraryRenameReq):
    old_name = os.path.basename(req.old_filename.strip())
    new_name = sanitize_filename(os.path.basename(req.new_filename.strip()))
    if not new_name:
        raise HTTPException(status_code=400, detail="Tên file mới không hợp lệ")

    if req.category == "download":
        if not new_name.lower().endswith(".mp4"):
            new_name += ".mp4"
        old_path = os.path.join(DOWNLOADS_DIR, old_name)
        new_path = os.path.join(DOWNLOADS_DIR, new_name)
        if not os.path.exists(old_path):
            raise HTTPException(status_code=404, detail="Không tìm thấy video gốc")
        if os.path.exists(new_path) and old_path != new_path:
            raise HTTPException(status_code=400, detail="Tên file mới đã tồn tại")
        os.rename(old_path, new_path)

        # Tự động đổi tên các file sub & âm thanh đi kèm để không bị mất liên kết
        old_stem = os.path.splitext(old_name)[0]
        new_stem = os.path.splitext(new_name)[0]
        for suffix in ["_zh.srt", "_vi.srt", "_bgm.wav", "_vocal.wav"]:
            s_old = os.path.join(DOWNLOADS_DIR, old_stem + suffix)
            s_new = os.path.join(DOWNLOADS_DIR, new_stem + suffix)
            if os.path.exists(s_old) and not os.path.exists(s_new):
                os.rename(s_old, s_new)
        return {"status": "ok", "new_filename": new_name}

    elif req.category == "output":
        if not new_name.lower().endswith(".mp4"):
            new_name += ".mp4"
        old_path = os.path.join(OUTPUTS_DIR, old_name)
        new_path = os.path.join(OUTPUTS_DIR, new_name)
        if not os.path.exists(old_path):
            raise HTTPException(status_code=404, detail="Không tìm thấy video xuất")
        if os.path.exists(new_path) and old_path != new_path:
            raise HTTPException(status_code=400, detail="Tên file mới đã tồn tại")
        os.rename(old_path, new_path)
        return {"status": "ok", "new_filename": new_name}

    elif req.category == "subtitle":
        if not new_name.lower().endswith(".srt"):
            new_name += ".srt"
        old_path = os.path.join(DOWNLOADS_DIR, old_name)
        new_path = os.path.join(DOWNLOADS_DIR, new_name)
        if not os.path.exists(old_path):
            raise HTTPException(status_code=404, detail="Không tìm thấy file phụ đề")
        if os.path.exists(new_path) and old_path != new_path:
            raise HTTPException(status_code=400, detail="Tên file mới đã tồn tại")
        os.rename(old_path, new_path)
        return {"status": "ok", "new_filename": new_name}

    raise HTTPException(status_code=400, detail="Loại file không hợp lệ")


@app.post("/api/library/delete")
async def delete_library_file(req: LibraryDeleteReq):
    fname = os.path.basename(req.filename.strip())
    if req.category == "download":
        target = os.path.join(DOWNLOADS_DIR, fname)
        if os.path.exists(target):
            os.remove(target)
        if req.delete_related:
            stem = os.path.splitext(fname)[0]
            for suffix in ["_zh.srt", "_vi.srt", "_bgm.wav", "_vocal.wav"]:
                rel = os.path.join(DOWNLOADS_DIR, stem + suffix)
                if os.path.exists(rel):
                    os.remove(rel)
        return {"status": "ok"}

    elif req.category == "output":
        target = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(target):
            os.remove(target)
        return {"status": "ok"}

    elif req.category == "subtitle":
        target = os.path.join(DOWNLOADS_DIR, fname)
        if os.path.exists(target):
            os.remove(target)
        return {"status": "ok"}

    raise HTTPException(status_code=400, detail="Loại file không hợp lệ")


@app.post("/api/bilibili/info")
async def api_bilibili_info(req: BiliInfoReq):
    try:
        bvid = parse_bvid(req.url)
        vinfo = get_video_info(bvid, cookie=req.cookie or "")
        pages = []
        for p in vinfo.get("pages", []):
            pages.append({
                "page": p.get("page", 1),
                "part": p.get("part", f"Tập {p.get('page', 1)}"),
                "cid": p.get("cid"),
                "duration_min": round(p.get("duration", 0) / 60.0, 1),
            })
        return {
            "bvid": bvid,
            "title": vinfo.get("title", bvid),
            "author": vinfo.get("owner", {}).get("name", "Unknown"),
            "pic": vinfo.get("pic", ""),
            "pages": pages,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/bilibili/download")
async def api_bilibili_download(req: BiliDownloadReq):
    tid = create_task("Tải video Bilibili")

    def worker():
        try:
            TASKS[tid]["message"] = "Đang lấy thông tin video và phân đoạn..."
            TASKS[tid]["progress"] = 10
            bvid = parse_bvid(req.url)
            vinfo = get_video_info(bvid, cookie=req.cookie or "")
            title = vinfo.get("title", bvid)
            pages = vinfo.get("pages", [])

            target_page = next((p for p in pages if p.get("page") == req.page), pages[0])
            cid = target_page["cid"]
            part_title = target_page.get("part", "")
            full_title = f"{title}_{part_title}" if part_title and part_title != title else title
            safe_title = sanitize_filename(full_title)

            final_filename = f"{safe_title}.mp4"
            final_output = os.path.join(DOWNLOADS_DIR, final_filename)

            # Kiểm tra phụ đề CC gốc của Bilibili
            subs = get_subtitles(bvid, cid, cookie=req.cookie or "")
            if subs:
                try:
                    sub_url = subs[0].get("subtitle_url", "")
                    if sub_url.startswith("//"):
                        sub_url = "https:" + sub_url
                    sub_data = make_api_request(sub_url)
                    srt_content = json_sub_to_srt(sub_data)
                    zh_srt_file = os.path.join(DOWNLOADS_DIR, f"{safe_title}_zh.srt")
                    with open(zh_srt_file, "w", encoding="utf-8") as f:
                        f.write(srt_content)
                except Exception:
                    pass

            TASKS[tid]["message"] = f"Đang lấy luồng media cho {safe_title}..."
            TASKS[tid]["progress"] = 25
            play_data = get_playurl(bvid, cid, qn=req.quality, prefer_dash=True, cookie=req.cookie or "")

            if "dash" in play_data:
                dash = play_data["dash"]
                v_streams = sorted(dash.get("video", []), key=lambda x: x.get("id", 0), reverse=True)
                a_streams = sorted(dash.get("audio", []), key=lambda x: x.get("id", 0), reverse=True)
                v_url = v_streams[0].get("baseUrl") or v_streams[0].get("base_url")
                a_url = (a_streams[0].get("baseUrl") or a_streams[0].get("base_url")) if a_streams else None

                tmp_v = os.path.join(DOWNLOADS_DIR, f"_temp_{bvid}_v.m4s")
                tmp_a = os.path.join(DOWNLOADS_DIR, f"_temp_{bvid}_a.m4s")

                TASKS[tid]["message"] = "Đang tải luồng Hình ảnh (Video Stream)..."
                TASKS[tid]["progress"] = 45
                download_with_curl(v_url, tmp_v, desc="Video")

                if a_url:
                    TASKS[tid]["message"] = "Đang tải luồng Âm thanh (Audio Stream)..."
                    TASKS[tid]["progress"] = 75
                    download_with_curl(a_url, tmp_a, desc="Audio")

                TASKS[tid]["message"] = "Đang ghép hình ảnh và âm thanh bằng FFmpeg..."
                TASKS[tid]["progress"] = 90
                cmd = [FFMPEG, "-y", "-i", tmp_v]
                if a_url:
                    cmd.extend(["-i", tmp_a, "-c:v", "copy", "-c:a", "copy"])
                else:
                    cmd.extend(["-c:v", "copy"])
                cmd.append(final_output)
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                if os.path.exists(tmp_v): os.remove(tmp_v)
                if os.path.exists(tmp_a): os.remove(tmp_a)
            elif "durl" in play_data:
                v_url = play_data["durl"][0]["url"]
                TASKS[tid]["message"] = "Đang tải file MP4..."
                TASKS[tid]["progress"] = 60
                download_with_curl(v_url, final_output, desc="MP4")

            TASKS[tid]["status"] = "done"
            TASKS[tid]["progress"] = 100
            TASKS[tid]["message"] = "Tải video hoàn tất!"
            TASKS[tid]["result"] = {
                "filename": final_filename,
                "url": f"/downloads/{urllib.parse.quote(final_filename)}",
            }
        except Exception as e:
            TASKS[tid]["status"] = "error"
            TASKS[tid]["error"] = str(e)
            TASKS[tid]["message"] = f"Lỗi: {e}"

    threading.Thread(target=worker, daemon=True).start()
    return {"task_id": tid}


@app.post("/api/colab/check")
async def api_colab_check(req: ColabCheckReq):
    try:
        url = req.colab_url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="Vui lòng nhập URL Colab")
        status = check_colab_worker(url)
        return {"status": "ok", "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể kết nối Colab: {e}")


@app.post("/api/extract-sub")
async def api_extract_sub(req: ExtractSubReq):
    video_path = os.path.join(DOWNLOADS_DIR, req.video_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file video")

    mode_title = "Colab GPU (Demucs + Whisper)" if req.colab_url else f"Local M1 (Whisper {req.model_size})"
    tid = create_task(f"Nhận diện giọng nói & tách nhạc nền [{mode_title}]")

    def worker():
        try:
            base, _ = os.path.splitext(video_path)
            output_srt = f"{base}_zh.srt"

            if req.colab_url and req.colab_url.strip():
                TASKS[tid]["message"] = f"Đang kết nối Cloud GPU Colab ({req.model_size} + Demucs v4)..."
                TASKS[tid]["progress"] = 30
            else:
                if req.separate_bgm:
                    TASKS[tid]["message"] = "Đang tách nhạc nền (BGM) & giọng nói bằng AI Spleeter ONNX..."
                    TASKS[tid]["progress"] = 25
                    separate_audio_stems(video_path)
                TASKS[tid]["message"] = f"Đang nhận diện phụ đề tiếng Trung (Faster-Whisper {req.model_size})..."
                TASKS[tid]["progress"] = 55

            srt_content = extract_subtitles(
                video_path,
                output_srt=output_srt,
                model_size=req.model_size,
                separate_bgm=req.separate_bgm,
                colab_url=req.colab_url,
            )
            TASKS[tid]["status"] = "done"
            TASKS[tid]["progress"] = 100
            TASKS[tid]["message"] = "Trích xuất phụ đề & BGM hoàn tất!"
            TASKS[tid]["result"] = {"zh_srt": srt_content}
        except Exception as e:
            TASKS[tid]["status"] = "error"
            TASKS[tid]["error"] = str(e)
            TASKS[tid]["message"] = f"Lỗi: {e}"

    threading.Thread(target=worker, daemon=True).start()
    return {"task_id": tid}


def google_translate_text(text: str, sl: str = "zh-CN", tl: str = "vi") -> str:
    """Dịch nhanh qua Google Translate Mobile (kế thừa từ pyvideotrans/_google.py)."""
    q = urllib.parse.quote(text)
    url = f"https://translate.google.com/m?sl={sl}&tl={tl}&hl={tl}&q={q}"
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        html_str = resp.read().decode("utf-8", errors="ignore")
    m = re.search(r'<div\s+class="result-container">([^<]+?)</div>', html_str)
    if m:
        import html
        return html.unescape(m.group(1).strip())
    return text


@app.post("/api/translate-auto")
async def api_translate_auto(req: AutoTranslateReq):
    blocks = re.split(r'\n\s*\n', req.srt_content.strip())
    if not blocks:
        raise HTTPException(status_code=400, detail="Nội dung phụ đề trống")

    if req.provider == "llm" and req.api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=req.api_key, base_url=req.api_url or "https://api.deepseek.com/v1")
            prompt = req.custom_prompt or (
                "Bạn là biên kịch phim tiên hiệp/kiếm hiệp chuyên nghiệp. "
                "Hãy dịch file phụ đề SRT tiếng Trung sau sang tiếng Việt chuẩn văn phong kiếm hiệp "
                "(xưng hô: ta - ngươi, cô nương, huynh đệ, bổn tọa, nữ đế...). "
                "Giữ nguyên chính xác số thứ tự và mốc thời gian timestamp. "
                "Câu từ gãy gọn, súc tích để đọc lồng tiếng khớp khẩu hình. Chỉ trả về nội dung SRT."
            )
            resp = client.chat.completions.create(
                model=req.model_name or "deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": req.srt_content},
                ],
                temperature=0.3,
            )
            out = resp.choices[0].message.content.strip()
            out = re.sub(r'^```[a-zA-Z]*\n', '', out)
            out = re.sub(r'\n```$', '', out)
            return {"vi_srt": out.strip()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Lỗi gọi LLM API: {e}")

    # Dịch bằng Google Translate theo cụm để nhanh và giữ ngữ cảnh
    parsed = []
    for b in blocks:
        lines = [l.strip() for l in b.strip().split("\n") if l.strip()]
        if len(lines) >= 3 and "-->" in lines[1]:
            parsed.append((lines[0], lines[1], " ".join(lines[2:])))
        elif len(lines) >= 2 and "-->" in lines[0]:
            parsed.append((str(len(parsed) + 1), lines[0], " ".join(lines[1:])))

    batch_size = 25
    translated_lines = []
    for i in range(0, len(parsed), batch_size):
        chunk = parsed[i:i + batch_size]
        joined_zh = "\n".join([item[2] for item in chunk])
        trans_chunk = google_translate_text(joined_zh).split("\n")
        for idx_in_chunk, item in enumerate(chunk):
            vi_text = trans_chunk[idx_in_chunk].strip() if idx_in_chunk < len(trans_chunk) else item[2]
            translated_lines.append(f"{item[0]}\n{item[1]}\n{vi_text}\n")

    return {"vi_srt": "\n".join(translated_lines)}


@app.post("/api/tts-preview")
async def api_tts_preview(req: PreviewVoiceReq):
    import edge_tts
    sample_text = req.text.strip() or "Không thể ngã xuống... Ta vẫn chưa thể chết ở nơi này! Cô nương, có thể phát ra âm thanh bình thường một chút được không?"
    out_name = f"preview_{uuid.uuid4().hex[:6]}.mp3"
    out_path = os.path.join(OUTPUTS_DIR, out_name)
    comm = edge_tts.Communicate(sample_text, req.voice, rate=req.rate, pitch=req.pitch)
    await comm.save(out_path)
    return {"url": f"/outputs/{out_name}"}


@app.post("/api/upload-logo")
async def api_upload_logo(file: UploadFile = File(...)):
    safe_name = sanitize_filename(file.filename or "custom_logo.png")
    dest = os.path.join(LOGOS_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": safe_name, "url": f"/logos/{urllib.parse.quote(safe_name)}"}


@app.post("/api/create-text-logo")
async def api_create_text_logo(req: TextLogoReq):
    from PIL import Image, ImageDraw
    text = req.text.strip() or "MY CHANNEL"
    width = max(160, len(text) * 14 + 36)
    height = 50
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(0, 0), (width - 1, height - 1)], radius=10, fill=(20, 20, 28, 200), outline=(0, 200, 255, 230), width=2)
    draw.text((18, 16), f"★ {text}", fill=(255, 255, 255, 245))
    fname = f"badge_{sanitize_filename(text)}.png"
    dest = os.path.join(LOGOS_DIR, fname)
    img.save(dest)
    return {"filename": fname, "url": f"/logos/{urllib.parse.quote(fname)}"}


@app.post("/api/dub-render")
async def api_dub_render(req: DubRenderReq):
    video_path = os.path.join(DOWNLOADS_DIR, req.video_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file video nguồn")

    if not req.vi_srt_content.strip():
        raise HTTPException(status_code=400, detail="Nội dung phụ đề Tiếng Việt đang trống!")

    base, _ = os.path.splitext(video_path)
    vi_srt_path = f"{base}_vi.srt"
    with open(vi_srt_path, "w", encoding="utf-8") as f:
        f.write(req.vi_srt_content.strip())

    v_stem = Path(video_path).stem
    out_filename = f"{v_stem}_dubbed_{int(time.time()) % 10000}.mp4"
    out_path = os.path.join(OUTPUTS_DIR, out_filename)

    logo_path = os.path.join(LOGOS_DIR, req.logo_filename) if req.logo_filename else None
    if logo_path and not os.path.exists(logo_path):
        logo_path = None

    tid = create_task("Lồng tiếng AI & Xuất Video")

    def worker():
        try:
            def progress_cb(msg: str, pct: int):
                TASKS[tid]["message"] = msg
                TASKS[tid]["progress"] = pct

            use_hardsub = req.burn_hardsub if req.burn_hardsub is not None else req.hardsub
            use_logo_pos = req.logo_position if req.logo_position else req.logo_pos

            asyncio.run(
                run_dubbing_pipeline(
                    video_path=video_path,
                    srt_path=vi_srt_path,
                    output_path=out_path,
                    voice=req.voice,
                    rate=req.rate,
                    pitch=req.pitch,
                    bgm_volume=req.bgm_volume,
                    separate_bgm=req.separate_bgm,
                    hardsub=use_hardsub,
                    logo_image=logo_path,
                    logo_pos=use_logo_pos,
                    logo_width=req.logo_width,
                    progress_cb=progress_cb,
                )
            )
            TASKS[tid]["status"] = "done"
            TASKS[tid]["progress"] = 100
            TASKS[tid]["message"] = "Xuất video hoàn chỉnh thành công!"
            out_url = f"/outputs/{urllib.parse.quote(out_filename)}"
            TASKS[tid]["result"] = {
                "filename": out_filename,
                "output_filename": out_filename,
                "url": out_url,
                "output_url": out_url,
            }
        except Exception as e:
            TASKS[tid]["status"] = "error"
            TASKS[tid]["error"] = str(e)
            TASKS[tid]["message"] = f"Lỗi: {e}"

    threading.Thread(target=worker, daemon=True).start()
    return {"task_id": tid}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🚀 BILIBILI AI SUB & DUBBING STUDIO ĐÃ KHỞI ĐỘNG!")
    print("👉 Truy cập ngay trên trình duyệt: http://127.0.0.1:8080")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8080)
