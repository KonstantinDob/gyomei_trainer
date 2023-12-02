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
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME)

.PHONY: style
style:
	git config --global --add safe.directory /workdir && pre-commit run --verbose --files gyomei_trainer/*

.PHONY: test-cov
test-cov:
	docker run --rm \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		pytest \
			-p no:logging \
			--cache-clear \
			--cov gyomei_trainer/builder \
			--cov gyomei_trainer/metrics \
			--cov gyomei_trainer/model \
			--cov gyomei_trainer/modules \
			--cov gyomei_trainer/state \
			--junitxml=pytest.xml \
			--cov-report term-missing:skip-covered \
			--cov-report xml:coverage.xml \
			tests | tee cov.txt
