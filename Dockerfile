ARG GPU_TAG=
ARG TF_VERSION=1.15.0
FROM tensorflow/tensorflow:${TF_VERSION}-gpu-py3
COPY requirements.txt .
RUN pip install -r requirements.txt
ADD src /app/src

