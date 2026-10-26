# 🎬 ToolMovieAI - Bilibili AI Sub & Dubbing Studio

Hệ thống tự động hóa tải video, trích xuất phụ đề tiếng Trung, dịch thuật kịch bản AI, lồng tiếng Tiếng Việt và xuất video chuẩn phim kiếm hiệp / tiên hiệp.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuocDung21/ToolMovieAI/blob/main/colab_gpu_worker.ipynb)
[![Kaggle GPU](https://img.shields.io/badge/Kaggle-Dual%20T4%20(30GB%20VRAM)-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/new)

---

## ⚡ Kiến trúc Hybrid: Local Mac M1 + Cloud GPU (Colab / Kaggle)

Để vừa tiết kiệm chi phí vừa đạt hiệu năng xử lý tối đa, hệ thống áp dụng kiến trúc **Hybrid Computing**:

- **🖥️ Local Mac (Apple Silicon M1/M2/M3)**:
  - Tải video Bilibili trực tiếp (sử dụng IP gia đình, chống bị Bilibili chặn datacenter/captcha).
  - Biên tập phụ đề, nghe thử giọng đọc và hòa âm âm lượng BGM.
  - Render Hardsub & Watermark bằng bộ giải mã/mã hóa phần cứng `h264_videotoolbox` cực nhanh và mát máy.
- **☁️ Cloud GPU (Google Colab hoặc Kaggle GPU Miễn phí)**:
  - **Demucs v4 (`htdemucs`)**: Tách nhạc nền và giọng thoại đạt chuẩn phòng thu.
  - **Faster-Whisper Large-v3 (CUDA FP16)**: Nhận diện phụ đề tiếng Trung siêu chính xác với tốc độ 15x – 20x realtime.
  - **Tối ưu Dual GPU (Kaggle T4 x2)**: Tự động chạy Demucs trên `cuda:0` và Whisper trên `cuda:1` song song!
  - Kết nối bảo mật 2 chiều qua **Cloudflare Tunnel (Zero-config)**, chỉ truyền tải file audio nhẹ (~15MB), không tốn băng thông video.

---

## 🚀 Hướng dẫn 1: Tăng tốc qua Google Colab (1-Click)

1. Mở file notebook trên Google Colab bằng cách click vào badge:  
   👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QuocDung21/ToolMovieAI/blob/main/colab_gpu_worker.ipynb)
2. Chọn **Runtime** > **Change runtime type** > Chọn **T4 GPU** > Lưu.
3. Nhấn **Run all**. Sau khi khởi động, Colab sẽ in ra một đường link:
   ```
   👉 COPY LINK NÀY DÁN VÀO WEB STUDIO TRÊN MAC CỦA BẠN:
      https://xxxx.trycloudflare.com
   ```
4. Mở Web Studio tại `http://127.0.0.1:8080`, tại **Bước 2: Tách & Dịch Phụ Đề**:
   - Dán đường link trên vào ô URL và nhấn **Kiểm tra kết nối**.
   - Bấm **Tách Sub Gốc**!

---

## 🚀 Hướng dẫn 2: Tăng tốc qua Kaggle (30h/tuần, Dual GPU T4 x2 & Móc GitHub)

1. Vào [kaggle.com](https://www.kaggle.com) > Bấm **Create** > **New Notebook**.
2. Nhìn sang bảng **Notebook Options** bên phải:
   - **Accelerator**: Chọn **GPU T4 x2** (30GB VRAM) hoặc **GPU P100**.
   - **Internet**: Chuyển sang **Internet on** (yêu cầu xác thực SĐT 1 lần).
3. Tải file `kaggle_gpu_worker.ipynb` từ máy lên hoặc copy mã nguồn vào Notebook.
4. Cell Bước 3 sẽ **tự động móc trực tiếp mã nguồn mới nhất từ GitHub repo**:
   - Nếu repo Private: Điền `GITHUB_TOKEN` của bạn.
   - Nếu repo Public: Để trống và chạy trực tiếp.
5. Chạy Notebook và copy link `https://xxxx.trycloudflare.com` dán vào Web Studio!

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
├── colab_worker.py          # FastAPI Worker server (hỗ trợ cả Single-GPU Colab & Dual-GPU Kaggle)
├── colab_gpu_worker.ipynb   # Jupyter Notebook 1-click triển khai trên Google Colab
├── kaggle_gpu_worker.ipynb  # Jupyter Notebook móc trực tiếp từ GitHub chạy trên Kaggle
├── bili_downloader.py       # Module tải video & audio Dash từ Bilibili
├── extract_sub.py           # Module nhận diện phụ đề (Whisper) & tách BGM (Demucs / Spleeter)
├── dub_video.py             # Module lồng tiếng Edge-TTS & render video hoàn chỉnh
├── web_app.py               # Máy chủ Web Studio (FastAPI)
├── templates/
│   └── index.html           # Giao diện Studio hiện đại (Dark theme)
└── start_studio.sh          # Script khởi chạy 1-click
```
