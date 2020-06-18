TF_VERSION=1.15.0
REBUILD=False

docker build --build-arg GPU_TAG="-gpu" --build-arg TF_VERSION=${TF_VERSION} -t attention-keras:tf-${TF_VERSION} .

if [ -z $(docker ps -lq --filter ancestor=attention-keras:tf-${TF_VERSION}) ]; then
	docker run -it --gpus all -v ${PWD}/src:/app/src -v ${PWD}/data:/app/data -v ${PWD}/results:/app/results --env-file .env \
	attention-keras:tf-${TF_VERSION} bash
else
	container_id = $(docker ps -lq --filter ancestor=attention-keras:tf-${TF_VERSION})
	docker exec -it ${container_id} bash
fi