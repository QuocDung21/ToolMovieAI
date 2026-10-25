# 🎬 ToolMovieAI - Bilibili AI Sub & Dubbing Studio

Hệ thống tự động hóa tải video, trích xuất phụ đề tiếng Trung, dịch thuật kịch bản AI, lồng tiếng Tiếng Việt và xuất video chuẩn phim kiếm hiệp / tiên hiệp.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuocDung21/ToolMovieAI/blob/main/colab_gpu_worker.ipynb)

---

## ⚡ Kiến trúc Hybrid: Local Mac M1 + Cloud GPU Colab

Để vừa tiết kiệm chi phí vừa đạt hiệu năng xử lý tối đa, hệ thống áp dụng kiến trúc **Hybrid Computing**:

- **🖥️ Local Mac (Apple Silicon M1/M2/M3)**:
  - Tải video Bilibili trực tiếp (sử dụng IP gia đình, chống bị Bilibili chặn datacenter/captcha).
  - Biên tập phụ đề, nghe thử giọng đọc và hòa âm âm lượng BGM.
  - Render Hardsub & Watermark bằng bộ giải mã/mã hóa phần cứng `h264_videotoolbox` cực nhanh và mát máy.
- **☁️ Cloud GPU (Google Colab T4 / L4 / A100 Miễn phí)**:
  - **Demucs v4 (`htdemucs`)**: Tách nhạc nền và giọng thoại đạt chuẩn phòng thu.
  - **Faster-Whisper Large-v3 (CUDA FP16)**: Nhận diện phụ đề tiếng Trung siêu chính xác với tốc độ 15x – 20x realtime.
  - Kết nối bảo mật 2 chiều qua **Cloudflare Tunnel (Zero-config)**, chỉ truyền tải file audio nhẹ (~15MB), không tốn băng thông video.

---

## 🚀 Hướng dẫn kích hoạt Tăng tốc GPU Colab (1-Click)

1. Mở file notebook trên Google Colab bằng cách click vào badge:  
   👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuocDung21/ToolMovieAI/blob/main/colab_gpu_worker.ipynb)
2. Chọn **Runtime** > **Change runtime type** > Chọn **T4 GPU** > Lưu.
3. Chạy các ô code từ Bước 1 đến Bước 3.
4. Sau khi khởi động, Colab sẽ in ra một đường link:
   ```
   👉 COPY LINK NÀY DÁN VÀO WEB STUDIO TRÊN MAC CỦA BẠN:
      https://xxxx.trycloudflare.com
   ```
5. Mở giao diện Web Studio tại `http://127.0.0.1:8080`, tại **Bước 2: Tách & Dịch Phụ Đề**:
   - Bật công tắc **Tăng tốc Cloud GPU (Google Colab)**.
   - Dán đường link trên vào ô URL và nhấn **Kiểm tra kết nối**.
   - Bấm **Tách Sub Gốc** để trải nghiệm tốc độ xử lý GPU cực nhanh!

---

## 🛠️ Cài đặt & Khởi chạy Local Studio

```bash
# Khởi động Web Studio
./start_studio.sh
# Hoặc chạy trực tiếp:
uvicorn web_app:app --host 127.0.0.1 --port 8080
```
Truy cập trình duyệt tại: **http://127.0.0.1:8080**

---

## 📁 Cấu trúc thư mục

```
ToolMovieAI/
├── colab_worker.py          # FastAPI Worker server chạy trên Google Colab GPU
├── colab_gpu_worker.ipynb   # Jupyter Notebook 1-click triển khai trên Google Colab
├── bili_downloader.py       # Module tải video & audio Dash từ Bilibili
├── extract_sub.py           # Module nhận diện phụ đề (Whisper) & tách BGM (Demucs / Spleeter)
├── dub_video.py             # Module lồng tiếng Edge-TTS & render video hoàn chỉnh
├── web_app.py               # Máy chủ Web Studio (FastAPI)
├── templates/
│   └── index.html           # Giao diện Studio hiện đại (Dark theme)
└── start_studio.sh          # Script khởi chạy 1-click
```
