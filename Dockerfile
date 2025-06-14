# 選用 Python 3.9 基礎映像檔
FROM python:3.9-slim

# 安裝系統相依套件（MediaPipe 需要的）
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製程式碼
COPY . .

# 安裝 Python 套件
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 啟動應用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
