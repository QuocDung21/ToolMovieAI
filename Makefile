# ==============================================================================
# ToolMovieAI - Automation Makefile
# ==============================================================================

PROJECT_DIR := $(CURDIR)
PYVIDEOTRANS_DIR := $(PROJECT_DIR)/pyvideotrans
HOST ?= 127.0.0.1
PORT ?= 8080

.PHONY: help run start dev stop restart status clean test

help:
	@echo ""
	@echo "🎬 \033[1;36mToolMovieAI - Management Commands\033[0m"
	@echo "--------------------------------------------------------"
	@echo "  \033[1;32mmake run\033[0m       Khởi chạy Web Studio tại http://$(HOST):$(PORT)"
	@echo "  \033[1;32mmake dev\033[0m       Chạy chế độ phát triển (Auto-reload code)"
	@echo "  \033[1;33mmake stop\033[0m      Dừng tiến trình Web Studio đang chạy trên port $(PORT)"
	@echo "  \033[1;33mmake restart\033[0m   Khởi động lại Web Studio"
	@echo "  \033[1;34mmake status\033[0m    Kiểm tra trạng thái máy chủ Web Studio"
	@echo "  \033[1;35mmake worker\033[0m    Chạy thử Worker server cục bộ (colab_worker.py)"
	@echo "  \033[1;31mmake clean\033[0m     Dọn dẹp file tạm (_temp_*), cache Python"
	@echo "  \033[1;34mmake test\033[0m      Kiểm tra cú pháp toàn bộ file Python"
	@echo "--------------------------------------------------------"
	@echo ""

run: start

start:
	@echo "🚀 Đang khởi động ToolMovieAI Web Studio..."
	@echo "🌐 Truy cập giao diện tại: http://$(HOST):$(PORT)"
	@if [ -d "$(PYVIDEOTRANS_DIR)" ]; then \
		cd "$(PYVIDEOTRANS_DIR)" && uv run --offline python "$(PROJECT_DIR)/web_app.py"; \
	else \
		python3 "$(PROJECT_DIR)/web_app.py"; \
	fi

dev:
	@echo "🛠️ Đang khởi động ToolMovieAI (Chế độ Dev Reload)..."
	@if [ -d "$(PYVIDEOTRANS_DIR)" ]; then \
		cd "$(PYVIDEOTRANS_DIR)" && uv run --offline uvicorn web_app:app --app-dir "$(PROJECT_DIR)" --host $(HOST) --port $(PORT) --reload; \
	else \
		uvicorn web_app:app --host $(HOST) --port $(PORT) --reload; \
	fi

stop:
	@echo "🛑 Đang dừng tiến trình Web Studio trên port $(PORT)..."
	@lsof -ti :$(PORT) | xargs kill -9 2>/dev/null || echo "ℹ️ Không có tiến trình nào đang chạy trên port $(PORT)."
	@echo "✅ Đã dừng máy chủ."

restart: stop start

status:
	@echo "🔍 Kiểm tra trạng thái máy chủ tại port $(PORT)..."
	@if lsof -i :$(PORT) >/dev/null 2>&1; then \
		echo "🟢 Máy chủ đang CHẠY trên port $(PORT):"; \
		lsof -i :$(PORT); \
	else \
		echo "🔴 Máy chủ hiện ĐANG TẮT."; \
	fi

worker:
	@echo "⚡ Khởi chạy GPU/Cloud Worker cục bộ trên port 8000..."
	@python3 colab_worker.py

clean:
	@echo "🧹 Đang dọn dẹp file tạm..."
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find downloads/ -name "_temp_*" -delete 2>/dev/null || true
	@find . -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✅ Dọn dẹp hoàn tất!"

test:
	@echo "🧪 Kiểm tra cú pháp toàn bộ file Python..."
	@python3 -m py_compile web_app.py extract_sub.py dub_video.py bili_downloader.py colab_worker.py
	@echo "✅ Toàn bộ file Python hợp lệ!"
