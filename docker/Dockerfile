FROM python:3.12

RUN --mount=type=cache,target=/root/.cache,sharing=private \
    apt update && \
    apt install -y libgl1-mesa-glx && \
    true

ADD . /vsr
ARG CUDA_VERSION=11.8
ARG HARDWARD_ACCELERATOR="cuda"

# 如果是 CUDA 版本，执行 CUDA 特定设置
RUN --mount=type=cache,target=/root/.cache,sharing=private \
    if [ "${HARDWARD_ACCELERATOR}" = "cuda" ]; then \
        pip install paddlepaddle==3.0 && \
        pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu$(echo ${CUDA_VERSION} | tr -d '.') && \
        pip install -r /vsr/requirements.txt; \
    fi

# 如果是 DirectML 版本，执行 DirectML 特定设置
RUN --mount=type=cache,target=/root/.cache,sharing=private \
    if [ "${HARDWARD_ACCELERATOR}" = "directml" ]; then \
        pip install paddlepaddle==3.0 && \
        pip install torch_directml==0.2.5.dev240914 && \
        pip install -r /vsr/requirements.txt; \
    fi

# 如果是 CPU 版本，执行 CPU 特定设置
RUN --mount=type=cache,target=/root/.cache,sharing=private \
    if [ "${HARDWARD_ACCELERATOR}" = "cpu" ]; then \
        pip install paddlepaddle==3.0 && \
        pip install -r /vsr/requirements.txt && \
        sed -i 's/HARDWARD_ACCELERATION_OPTION *= *.*/HARDWARD_ACCELERATION_OPTION = False/g' /vsr/backend/config.py; \
    fi

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/
WORKDIR /vsr
CMD ["python", "/vsr/backend/main.py"]