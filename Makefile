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
	docker run --rm -dit \
		--name=$(NAME) \
		$(NAME)

.PHONY: style
style:
	git config --global --add safe.directory /workdir && pre-commit run --verbose --files gyomei_trainer/*
