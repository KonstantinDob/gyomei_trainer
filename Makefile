NAME?=trainer
GPUS?=all

.PHONY: build
build:
	docker build -t $(NAME) .

.PHONY: stop
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

.PHONY: run
run:
	docker run --rm -it \
		--gpus=$(GPUS) \
		--name=$(NAME) \
		$(NAME)
