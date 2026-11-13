#!/usr/bin/env python3

# Markdown-таблицы для отчёта из results.csv (вывод ./hw_bench)
# Для каждого числа потоков:         таблицы "Scalar -> SIMD (ускорение)" по размерам данных и типам
# Для замеров на нескольких потоках: таблица масштабирования по потокам (SIMD, максимальный размер)
"""
Использование:
  python3 make_report_tables.py results.csv > tables.md           # таблицы в stdout
  python3 make_report_tables.py results.csv --inject REPORT.md    # вставить таблицы в отчёт между маркерами
      <!-- TABLES:BEGIN -->  ...  <!-- TABLES:END -->
"""
import csv
import io
import sys

args = sys.argv[1:]
inject = None
if "--inject" in args:
    i = args.index("--inject")
    inject = args[i + 1]
    del args[i:i + 2]
path = args[0] if args else "results.csv"
_real_stdout = sys.stdout
if inject:
    sys.stdout = io.StringIO()
rows = list(csv.DictReader(open(path)))

d = {}       # (op, type, size, backend, threads) -> value
metric = {}  # op -> единица измерения
for r in rows:
    d[(r["op"], r["type"], int(r["size"]), r["backend"], int(r["threads"]))] = float(r["value"])
    metric[r["op"]] = r["metric"]

threads = sorted({k[4] for k in d})
types = [t for t in ("int32", "float32", "float64") if any(k[1] == t for k in d)]
elem_ops = [o for o in ("add", "sub", "mul", "div", "sum", "mean", "silu", "gelu") if o in metric]


def size_label(n):
    return f"2^{n.bit_length() - 1}" if n & (n - 1) == 0 else str(n)


def pair(op, t, s, th):
    sc, si = d[(op, t, s, "scalar", th)], d[(op, t, s, "simd", th)]
    return f"{sc:.2f} → {si:.2f} (×{si / sc:.2f})"


for th in threads:
    print(f"## Потоков: {th}\n")
    print("Поэлементные операции, reduction и активации. Значение: Scalar → SIMD (ускорение) "
          "Единицы: GB/s (поэлементные операции, reduction) и Gelem/s (silu, gelu)\n")
    for t in types:
        sizes = sorted({k[2] for k in d if k[0] == "add" and k[1] == t})
        print(f"**{t}**\n")
        print("| операция | " + " | ".join(size_label(s) + " эл." for s in sizes) + " |")
        print("|---|" + "---|" * len(sizes))
        for op in elem_ops:
            print(f"| {op} ({metric[op]}) | " + " | ".join(pair(op, t, s, th) for s in sizes) + " |")
        print()
    if "matmul" in metric:
        print("Матричное умножение N x N, GFLOP/s: Scalar → SIMD (ускорение)\n")
        sizes = sorted({k[2] for k in d if k[0] == "matmul"})
        print("| тип | " + " | ".join(f"N={s}" for s in sizes) + " |")
        print("|---|" + "---|" * len(sizes))
        for t in types:
            print(f"| {t} | " + " | ".join(pair("matmul", t, s, th) for s in sizes) + " |")
        print()

if len(threads) > 1:
    print("## Масштабирование по потокам (SIMD, максимальный размер данных)\n")
    print("Значение и ускорение относительно 1 потока\n")
    for t in types:
        print(f"**{t}**\n")
        print("| операция | " + " | ".join(f"{th} пот." for th in threads) + " |")
        print("|---|" + "---|" * len(threads))
        for op in elem_ops + (["matmul"] if "matmul" in metric else []):
            big = max(k[2] for k in d if k[0] == op and k[1] == t)
            base = d[(op, t, big, "simd", threads[0])]
            cells = [f"{d[(op, t, big, 'simd', th)]:.2f} (×{d[(op, t, big, 'simd', th)] / base:.2f})" for th in threads]
            label = f"{op} ({metric[op]}, N={big})" if op == "matmul" else f"{op} ({metric[op]}, {size_label(big)})"
            print(f"| {label} | " + " | ".join(cells) + " |")
        print()

if inject:
    tables = sys.stdout.getvalue().rstrip() + "\n"
    sys.stdout = _real_stdout
    begin, end = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
    text = open(inject, encoding="utf-8").read()
    a, b = text.find(begin), text.find(end)
    if a < 0 or b < 0 or b < a:
        sys.exit(f"markers {begin} / {end} not found in {inject}")
    open(inject, "w", encoding="utf-8").write(text[:a + len(begin)] + "\n\n" + tables + "\n" + text[b:])
    print(f"tables injected into {inject}")
