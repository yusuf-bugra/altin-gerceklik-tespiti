FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
COPY Base-RCNN-FPN.yaml ./

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir opencv-python flask gdown
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/fvcore.git
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git

EXPOSE 7860

CMD ["python", "app.py"]

RUN mkdir -p /app/model /app/uploads && chmod -R 777 /app/model /app/uploads
ENV MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p /app/model /app/uploads /tmp/matplotlib /tmp/gdown && chmod -R 777 /app/model /app/uploads /tmp
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV GDOWN_CACHE_DIR=/tmp/gdown

EXPOSE 7860
