IMAGE ?= mm-sae:suite
TARGET ?= cpu
GPU_FLAGS = $(if $(filter gpu,$(TARGET)),--gpus all,)
DATA ?= $(CURDIR)/data
CACHE ?= $(CURDIR)/cache
RUNS ?= $(CURDIR)/runs
ARGS ?= run configs/paper.yaml
SAE ?=
MOUNTS = -v "$(DATA):/workspace/data:ro" -v "$(CACHE):/workspace/cache" -v "$(RUNS):/workspace/runs"
ifneq ($(strip $(SAE)),)
MOUNTS += -v "$(SAE):/weights:ro"
REUSE = --sae /weights
endif
.PHONY: build smoke run
build:
	docker build --target $(TARGET) -f docker/Dockerfile -t $(IMAGE) .
run:
	@mkdir -p "$(DATA)" "$(CACHE)" "$(RUNS)"
	docker run --rm $(GPU_FLAGS) $(MOUNTS) $(IMAGE) $(ARGS) $(REUSE)
smoke:
	$(MAKE) run ARGS="run configs/smoke.yaml"
