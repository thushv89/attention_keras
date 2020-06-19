TF_VERSION=1.15.2
GPU_TAG="-gpu" # Set this -gpu for the GPU version or to empty if you want the CPU version

docker build --build-arg GPU_TAG=${GPU_TAG} --build-arg TF_VERSION=${TF_VERSION} -t attention-keras:tf-${TF_VERSION}${GPU_TAG} .

if [ -z $(docker ps -lq --filter ancestor=attention-keras:tf-${TF_VERSION}${GPU_TAG}) ]; then
	docker run -it --gpus all -v ${PWD}/src:/app/src -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results --env-file .env \
	attention-keras:tf-${TF_VERSION}${GPU_TAG} bash
else
	container_id = $(docker ps -lq --filter ancestor=attention-keras:tf-${TF_VERSION}${GPU_TAG})
	docker exec -it ${container_id} bash
fi