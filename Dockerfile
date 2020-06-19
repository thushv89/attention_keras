ARG GPU_TAG=
ARG TF_VERSION=1.15.2
FROM tensorflow/tensorflow:${TF_VERSION}${GPU_TAG}-py3
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV PYTHONPATH=/app/src
ADD src /app/src
WORKDIR app/

