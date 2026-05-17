FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y --fix-missing \
    git \
    gcc \
    g++ \
    libosmesa6-dev \
    libgl1-mesa-glx \
    patchelf \
    libglew-dev \
    glew-utils \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY ./download/mujoco210-linux-x86_64.tar.gz /tmp/mujoco.tar.gz
RUN mkdir -p /root/.mujoco \
    && tar -xf /tmp/mujoco.tar.gz -C /root/.mujoco \
    && rm /tmp/mujoco.tar.gz

RUN pip install --no-cache-dir \
    gym==0.23.1 \
    stable-baselines3==2.4.0 \
    wandb \
    h5py==3.8.0 \
    imageio \
    tqdm \
    seaborn \
    plotly \
    prettytable

RUN pip install --no-cache-dir \
    mujoco_py==2.1.2.14 \
    Cython==0.29.32 \
    dm_control==1.0.13 \
    && pip install --no-cache-dir \
    git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

COPY submodule/ ./submodule/
COPY layer/ ./layer/
COPY utilities/ ./utilities/
COPY log/ ./log/
COPY main.py .

RUN mkdir -p /root/.d4rl/datasets  /app/weight

ENV D4RL_SUPPRESS_IMPORT_ERROR=1
ENV MUJOCO_GL=egl
ENV MUJOCO_KEY_PATH=/root/.mujoco
ENV MJLIB_PATH=/root/.mujoco/mujoco210/lib/libmujoco.so.2.1.1
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/root/.mujoco/mujoco210/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/app

RUN echo '#!/bin/bash\n\
echo "=== GPU Test ==="\n\
nvidia-smi\n\
echo "=== PyTorch CUDA Test ==="\n\
python -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA Available: {torch.cuda.is_available()}\"); print(f\"GPU Count: {torch.cuda.device_count()}\")"\n\
exec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py"]

EXPOSE 6006 8888