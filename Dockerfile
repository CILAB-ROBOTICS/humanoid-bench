# CUDA 12.2 런타임 기반 이미지
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 시스템 필수 패키지 설치 (mujoco 실행에 필요한 패키지 포함)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    patchelf \
    ca-certificates \
    python3-dev \
    libx11-dev \
    libxcb1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libfontconfig1 \
    nano \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 아나콘다 설치
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh -O ~/anaconda.sh \
    && bash ~/anaconda.sh -b -p /opt/conda \
    && rm ~/anaconda.sh

# 아나콘다 환경 설정
ENV PATH="/opt/conda/bin:$PATH"

# Python 3.11 설치
RUN conda install -y python=3.11 && conda clean -a -y

# 작업 디렉토리 설정
WORKDIR /workspace

# 코드 및 requirements 복사
COPY humanoid_bench .
COPY README.md .
COPY setup.py .
# HumanoidBench 설치
RUN pip install -e .

COPY requirements_jaxrl.txt .
COPY requirements_dreamer.txt .
COPY requirements_tdmpc.txt .

# requirements 파일들 개별 설치 (레이어 분리)
RUN pip install -r requirements_jaxrl.txt
RUN pip install -r requirements_dreamer.txt
RUN pip install -r requirements_tdmpc.txt
RUN pip install opencv-python-headless

# CUDA 호환 패키지 추가 설치 (호스트와 버전 맞추기 위해)
RUN apt-get update && apt-get install -y cuda-compat-12-2 && rm -rf /var/lib/apt/lists/*

# JAX GPU 버전 설치
RUN pip install --upgrade pip && \
    pip install "jax[cuda12]==0.4.28" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html orbax-checkpoint

# 환경 변수 설정 (mujoco 관련 경로)
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
ENV LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
ENV MUJOCO_GL=osmesa
# 기본 커맨드
CMD ["bash"]
