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

.PHONY: _test-cov
test-cov:
	pytest \
		no:logging \
		--cache-clear \
		--cov gyomei_trainer/builder \
		--cov gyomei_trainer/metrics \
		--cov gyomei_trainer/model \
		--cov gyomei_trainer/modules \
		--cov gyomei_trainer/state \
		tests
