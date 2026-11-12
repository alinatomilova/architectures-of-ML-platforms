#   make            -> Release build
#   make bench      -> build + run benchmark, save to results.csv
#   make report     -> generate tables.md from results.csv
#   make all-run    -> bench + report
#   make clean

BUILD_DIR   ?= build
BUILD_TYPE  ?= Release
JOBS        ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CMAKE       ?= cmake
PYTHON      ?= python3

.PHONY: all configure build bench report all-run clean rebuild help

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)

bench: build
	@echo ">>> running benchmark -> results.csv (stderr -> bench.log)"
	./$(BUILD_DIR)/hw_bench > results.csv 2> bench.log
	@echo ">>> done: $$(wc -l < results.csv) lines in results.csv"

report:
	@if [ ! -f results.csv ]; then echo "results.csv not found: run 'make bench' first"; exit 1; fi
	$(PYTHON) make_report_tables.py results.csv > tables.md
	@echo ">>> tables.md written"

all-run: bench report

clean:
	rm -rf $(BUILD_DIR) tables.md

rebuild: clean build

help:
	@echo "Targets:"
	@echo "  all (default)  configure + build Release"
	@echo "  bench          build + run hw_bench > results.csv"
	@echo "  report         make_report_tables.py results.csv > tables.md"
	@echo "  all-run        bench + report"
	@echo "  clean          rm -rf build tables.md"