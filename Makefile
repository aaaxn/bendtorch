# Shortcuts for the commands in README.md and AGENTS.md. Each gate writes its
# evidence to a new directory, .evidence/<target>-$(RUN); set RUN to pick one.

RUN  ?= $(shell date +%Y%m%d-%H%M%S)
PY   := uv run --locked python
BEND := bun $(CURDIR)/.tools/bend/bend2/main.ts
PYDIRS := tools tests verification dev

export BEND_NO_TELEMETRY := 1
export BENDTT := $(CURDIR)/.tools/kernel/bendtt
export LEAN_STACK_SIZE_KB := 4194304

.PHONY: setup check proofs test gates verify package mnist

setup:            # the pinned Bend2, BendTT, Lean and Clang, in .tools/
	uv sync --locked
	$(PY) -m tools.bootstrap

check: package    # seconds: lint, types, the package and one quick test
	uv run --locked ruff check $(PYDIRS)
	uv run --locked ruff format --check $(PYDIRS)
	uv run --locked ty check
	$(PY) -m unittest discover -s tests -p test_tensor_numerics.py

proofs:           # every law, by the Bend checker and the BendTT kernel
	$(BEND) PROOF.bend --verdict

test:             # every unit test, each module in its own process
	$(PY) -m tools.test

gates:            # the suite gates, then the contracts and the families
	$(PY) -m tools.check all --output .evidence/check-$(RUN)
	$(PY) -m verification.contracts.run --output .evidence/contracts-$(RUN)
	$(PY) -m verification.families.run --output .evidence/families-$(RUN)

verify: check proofs test gates   # what a release needs to pass

package:          # the files `bend bendtorch.bend --publish` would ship
	$(PY) -m tools.package

mnist:            # build the MNIST example natively; docs/mnist.md runs it
	mkdir -p .evidence/mnist-$(RUN)
	$(BEND) examples/mnist/train.bend -o .evidence/mnist-$(RUN)/train.c
	$(CURDIR)/.tools/llvm/usr/bin/clang-19 -std=c11 -O3 \
	  .evidence/mnist-$(RUN)/train.c -lpthread -lm -o .evidence/mnist-$(RUN)/train
	@echo built .evidence/mnist-$(RUN)/train
