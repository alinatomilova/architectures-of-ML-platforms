#!/usr/bin/env python3
"""python3 make_report_tables.py [results.csv] > tables.md"""
import csv, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
rows = list(csv.DictReader(open(path)))
d = {(r["op"], r["type"], int(r["size"]), r["backend"], int(r["threads"])): float(r["value"]) for r in rows}
threads = sorted({k[4] for k in d})
types = ["int32", "float32", "float64"]
elem_ops = ["add", "sub", "mul", "div", "sum", "mean"]


def size_label(n):
    return f"2^{n.bit_length() - 1}"


def cell(op, t, s, th):
    sc, si = d[(op, t, s, "scalar", th)], d[(op, t, s, "simd", th)]
    return f"{sc:.1f} --> {si:.1f} (x{si / sc:.2f})"


for th in threads:
    print(f"# Количество потоков: {th}\n")
    print("### Поэлементные операции и reduction, GB/s: Scalar --> SIMD (ускорение)\n")
    for t in types:
        sizes = sorted({k[2] for k in d if k[0] == "add" and k[1] == t})
        print(f"**{t}**\n")
        print("| операция | " + " | ".join(size_label(s) + " эл." for s in sizes) + " |")
        print("|---|" + "---|" * len(sizes))
        for op in elem_ops:
            print(f"| {op} | " + " | ".join(cell(op, t, s, th) for s in sizes) + " |")
        print()
    print("### Матричное умножение N x N, GFLOP/s: Scalar --> SIMD (ускорение)\n")
    sizes = sorted({k[2] for k in d if k[0] == "matmul"})
    print("| тип | " + " | ".join(f"N={s}" for s in sizes) + " |")
    print("|---|" + "---|" * len(sizes))
    for t in types:
        print(f"| {t} | " + " | ".join(cell("matmul", t, s, th) for s in sizes) + " |")
    print()
