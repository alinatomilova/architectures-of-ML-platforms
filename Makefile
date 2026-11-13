#   make            -> Release build (hw_tests + hw_bench)
#   make test       -> build + run correctness tests
#   make bench      -> build + run benchmark, save to results.csv
#   make report     -> generate tables.md from results.csv
#   make inject     -> insert the tables into REPORT.md (between TABLES markers)
#   make specs      -> print machine info for the "stand" section of the report
#   make all-run    -> test + bench + inject
#   make clean

BUILD_DIR   ?= build
BUILD_TYPE  ?= Release
JOBS        ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CMAKE       ?= cmake
PYTHON      ?= python3

.PHONY: all configure build test bench report inject specs all-run clean rebuild help

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)

test: build
	@echo ">>> running correctness tests (expect: '... checks, 0 failures')"
	./$(BUILD_DIR)/hw_tests

bench: build
	@echo ">>> running benchmark -> results.csv (stderr -> bench.log); takes several minutes"
	./$(BUILD_DIR)/hw_bench > results.csv 2> bench.log
	@echo ">>> done: $$(wc -l < results.csv) lines in results.csv"
	@cat bench.log

report:
	@if [ ! -f results.csv ]; then echo "results.csv not found: run 'make bench' first"; exit 1; fi
	$(PYTHON) make_report_tables.py results.csv > tables.md
	@echo ">>> tables.md written"

inject:
	@if [ ! -f results.csv ]; then echo "results.csv not found: run 'make bench' first"; exit 1; fi
	$(PYTHON) make_report_tables.py results.csv --inject REPORT.md

specs:
	@echo "== CPU =="
	@lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread|L1d|L2|L3' || sysctl -n machdep.cpu.brand_string 2>/dev/null || true
	@echo "SIMD flags: $$(grep -o -w -E 'avx|avx2|avx512f|fma' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' ')"
	@echo "== Memory =="
	@(free -h 2>/dev/null | head -2) || true
	@echo "== OS / toolchain =="
	@(. /etc/os-release 2>/dev/null && echo "$$PRETTY_NAME") || uname -sr
	@(g++ --version 2>/dev/null || c++ --version 2>/dev/null) | head -1
	@$(CMAKE) --version | head -1

all-run: test bench inject

clean:
	rm -rf $(BUILD_DIR) tables.md

rebuild: clean build

help:
	@echo "Targets:"
	@echo "  all (default)  configure + build Release (hw_tests, hw_bench)"
	@echo "  test           build + run hw_tests (correctness)"
	@echo "  bench          build + run hw_bench > results.csv (stderr -> bench.log)"
	@echo "  report         make_report_tables.py results.csv > tables.md"
	@echo "  inject         insert the tables into REPORT.md"
	@echo "  specs          print CPU / memory / compiler info for the report"
	@echo "  all-run        test + bench + inject"
	@echo "  clean          rm -rf build tables.md"
	@echo "Variables: BUILD_DIR, BUILD_TYPE, JOBS, CMAKE, PYTHON"
