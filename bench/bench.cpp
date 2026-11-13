// Нагрузочные тесты (замер производительности)
// op: add, sub, mul, div, sum, mean, silu, gelu, matmul
// type: int32, float32, float64
// backend: scalar, simd
// threads: 1, 2, 4, ... max_threads
// metric: GB/s (поэлементные операции/reduce), Gelem/s (SiLU/GELU), GFLOP/s (matmul)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "../include/activations.hpp"

using Clock = std::chrono::steady_clock;

template <class T>
std::vector<T> make_data(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<T> v(n);
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<int> d(1, 9);  // без нулей (деление) и без переполнения сумм
        for (auto& x : v) x = static_cast<T>(d(rng));
    } else {
        std::uniform_real_distribution<double> d(0.5, 1.5);
        for (auto& x : v) x = static_cast<T>(d(rng));
    }
    return v;
}

// медиана времени (мс)
double time_ms(const std::function<void()>& fn) {
    fn();  // прогрев
    std::vector<double> t;
    double total = 0;
    while (t.size() < 5 || (total < 300.0 && t.size() < 200)) {
        auto a = Clock::now();
        fn();
        auto b = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(b - a).count();
        t.push_back(ms);
        total += ms;
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

struct Csv {
    void row(const char* op, const char* type, std::size_t size, const char* backend, std::size_t threads,
             double ms, const char* metric, double value) {
        std::printf("%s,%s,%zu,%s,%zu,%.5f,%s,%.4f\n", op, type, size, backend, threads, ms, metric, value);
        std::fflush(stdout);
    }
};

template <class T>
const char* type_name() {
    if constexpr (std::is_same_v<T, std::int32_t>) return "int32";
    else if constexpr (std::is_same_v<T, float>) return "float32";
    else return "float64";
}

template <class B, class T>
void bench_backend(Csv& csv, std::size_t threads, std::size_t n, const std::vector<T>& a,
                   const std::vector<T>& b, std::vector<T>& out) {
    const char* tn = type_name<T>();
    const char* bn = B::name;
    volatile double sink = 0;
    auto elementwise = [&](const char* op, auto fn) {
        double ms = time_ms(fn);
        csv.row(op, tn, n, bn, threads, ms, "GB/s", 3.0 * n * sizeof(T) / (ms * 1e6));
    };
    elementwise("add", [&] { hw::add<B>(a.data(), b.data(), out.data(), n, threads); });
    elementwise("sub", [&] { hw::sub<B>(a.data(), b.data(), out.data(), n, threads); });
    elementwise("mul", [&] { hw::mul<B>(a.data(), b.data(), out.data(), n, threads); });
    elementwise("div", [&] { hw::div<B>(a.data(), b.data(), out.data(), n, threads); });
    double ms = time_ms([&] { sink = sink + static_cast<double>(hw::sum<B>(a.data(), n, threads)); });
    csv.row("sum", tn, n, bn, threads, ms, "GB/s", 1.0 * n * sizeof(T) / (ms * 1e6));
    ms = time_ms([&] { sink = sink + hw::mean<B>(a.data(), n, threads); });
    csv.row("mean", tn, n, bn, threads, ms, "GB/s", 1.0 * n * sizeof(T) / (ms * 1e6));
    // активации: для int32 вход приводится к float, результат float
    std::vector<hw::ActOut<T>> act_out(n);
    ms = time_ms([&] { hw::silu<B>(a.data(), act_out.data(), n, threads); });
    csv.row("silu", tn, n, bn, threads, ms, "Gelem/s", n / (ms * 1e6));
    ms = time_ms([&] { hw::gelu<B>(a.data(), act_out.data(), n, threads); });
    csv.row("gelu", tn, n, bn, threads, ms, "Gelem/s", n / (ms * 1e6));
}

template <class B, class T>
void bench_matmul(Csv& csv, std::size_t threads, std::size_t N, const std::vector<T>& A,
                  const std::vector<T>& Bm, std::vector<T>& C) {
    double ms = time_ms([&] { hw::matmul<B>(A.data(), Bm.data(), C.data(), N, N, N, threads); });
    csv.row("matmul", type_name<T>(), N, B::name, threads, ms, "GFLOP/s", 2.0 * N * N * N / (ms * 1e6));
}

template <class T>
void run_type(Csv& csv, const std::vector<std::size_t>& thread_list) {
    for (std::size_t n : {std::size_t{1} << 10, std::size_t{1} << 14, std::size_t{1} << 18, std::size_t{1} << 22,
                          std::size_t{1} << 24}) {
        auto a = make_data<T>(n, 11), b = make_data<T>(n, 12);
        std::vector<T> out(n);
        for (std::size_t th : thread_list) {
            bench_backend<hw::Scalar>(csv, th, n, a, b, out);
            bench_backend<hw::Simd>(csv, th, n, a, b, out);
        }
    }
    for (std::size_t N : {64, 128, 256, 512, 1024}) {
        auto A = make_data<T>(N * N, 21), Bm = make_data<T>(N * N, 22);
        std::vector<T> C(N * N);
        for (std::size_t th : thread_list) {
            bench_matmul<hw::Scalar>(csv, th, N, A, Bm, C);
            bench_matmul<hw::Simd>(csv, th, N, A, Bm, C);
        }
    }
}

int main() {
    std::size_t maxt = hw::ThreadPool::instance().max_threads();
    std::vector<std::size_t> thread_list;
    for (std::size_t t = 1; t < maxt; t *= 2) thread_list.push_back(t);
    thread_list.push_back(maxt);
    std::fprintf(stderr, "simd bytes: %zu, max threads: %zu\n", hw::kSimdBytes, maxt);
    std::printf("op,type,size,backend,threads,time_ms,metric,value\n");
    Csv csv;
    run_type<std::int32_t>(csv, thread_list);
    run_type<float>(csv, thread_list);
    run_type<double>(csv, thread_list);
    std::fprintf(stderr, "done\n");
}
