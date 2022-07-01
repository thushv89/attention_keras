ARG GPU_TAG=
ARG TF_VERSION=2.9.1
FROM tensorflow/tensorflow:${TF_VERSION}${GPU_TAG}
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV PYTHONPATH=/app/src
WORKDIR app/

