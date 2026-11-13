## Домашнее задание No 1

Реализовать:
1. Функции арифметических операций(`+`, `-`, `*`, `/`) для массива чисел(Scalar, SIMD).
2. Функции reduction для элементов массива: суммы элементов и среднего значения(Scalar, SIMD).
3. Функции активации `SiLU`, `GELU` (аппроксимации)(Scalar, SIMD).
4. Функции матричного умножения (Scalar, SIMD).
5. п.1-4 реализовать с использованием обобщенного подхода(шаблоны С++, callable объекты) с минимальным дублированием кода. 
6. Использовать параллелизацию, можно с использованием сторонних библиотек(parallel_for, parallel_reduce).
7. Тесты на корректность(можно использовать Scalar реализацию как опорную).
8. Нагрузочные тесты для двух реализаций(Scalar, SIMD), для разного размера данных и типов данных.

Типы данных: int32, float32, float64.

**Отчет: `REPORT.md`**

### Запуск тестов

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure        # тесты на корректность (./build/hw_tests)
./build/hw_bench > results.csv 2> bench.log       # нагрузочные тесты
python3 make_report_tables.py results.csv --inject REPORT.md   # пересобрать таблицы
```