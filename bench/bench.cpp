// Замер производительности
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "include/ops.hpp"

using Clock = std::chrono::steady_clock;

template <class T>
std::vector<T> make_data(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<T> v(n);
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<int> d(1, 9);
        for (auto& x : v) x = static_cast<T>(d(rng));
    } else {
        std::uniform_real_distribution<double> d(0.5, 1.5);
        for (auto& x : v) x = static_cast<T>(d(rng));
    }
    return v;
}

// медиана времени (мс)
double time_ms(const std::function<void()>& fn) {
    fn();
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

// сравнение значений, посчитанных из данных типа T
template <class T, class V>
bool close_as(V a, V b) {
    if constexpr (std::is_integral_v<T>) return double(a) == double(b);
    double tol = std::is_same_v<T, float> ? 1e-4 : 1e-10;
    return std::fabs(double(a) - double(b)) <= tol * std::max(1.0, std::fabs(double(b)));
}
template <class T>
bool close(T a, T b) { return close_as<T>(a, b); }

template <class T>
void check_vec(const std::vector<T>& x, const std::vector<T>& ref, const char* what) {
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!close(x[i], ref[i])) {
            std::fprintf(stderr, "MISMATCH in %s at %zu: %g vs %g\n", what, i, double(x[i]), double(ref[i]));
            std::exit(1);
        }
}

template <class T>
void sanity_check() {
    const std::size_t n = 200003;  // нечётный размер для проверки хвостов и разбиений на потоки
    auto a = make_data<T>(n, 1), b = make_data<T>(n, 2);
    std::vector<T> ref(n), out(n);
    auto each = [&](auto fn, const char* name) {
        fn(hw::Scalar{}, ref.data(), std::size_t{1});
        fn(hw::Simd{}, out.data(), std::size_t{1});
        check_vec(out, ref, name);
        fn(hw::Simd{}, out.data(), std::size_t{0});
        check_vec(out, ref, name);
    };
    each([&](auto B, T* o, std::size_t th) { hw::add<decltype(B)>(a.data(), b.data(), o, n, th); }, "add");
    each([&](auto B, T* o, std::size_t th) { hw::sub<decltype(B)>(a.data(), b.data(), o, n, th); }, "sub");
    each([&](auto B, T* o, std::size_t th) { hw::mul<decltype(B)>(a.data(), b.data(), o, n, th); }, "mul");
    each([&](auto B, T* o, std::size_t th) { hw::div<decltype(B)>(a.data(), b.data(), o, n, th); }, "div");

    T s0 = hw::sum<hw::Scalar>(a.data(), n, 1);
    if (!close(hw::sum<hw::Simd>(a.data(), n, 1), s0) || !close(hw::sum<hw::Simd>(a.data(), n, 0), s0)) {
        std::fprintf(stderr, "MISMATCH in sum\n");
        std::exit(1);
    }
    if (!close_as<T>(hw::mean<hw::Simd>(a.data(), n, 0), hw::mean<hw::Scalar>(a.data(), n, 1))) {
        std::fprintf(stderr, "MISMATCH in mean\n");
        std::exit(1);
    }

    const std::size_t M = 67, K = 53, N = 71;  // не кратны MR и ширине вектора
    auto A = make_data<T>(M * K, 3), Bm = make_data<T>(K * N, 4);
    std::vector<T> c0(M * N), c1(M * N), c2(M * N);
    hw::matmul<hw::Scalar>(A.data(), Bm.data(), c0.data(), M, K, N, 1);
    hw::matmul<hw::Simd>(A.data(), Bm.data(), c1.data(), M, K, N, 1);
    hw::matmul<hw::Simd>(A.data(), Bm.data(), c2.data(), M, K, N, 0);
    check_vec(c1, c0, "matmul simd");
    check_vec(c2, c0, "matmul simd parallel");
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
}

template <class B, class T>
void bench_matmul(Csv& csv, std::size_t threads, std::size_t N, const std::vector<T>& A,
                  const std::vector<T>& Bm, std::vector<T>& C) {
    double ms = time_ms([&] { hw::matmul<B>(A.data(), Bm.data(), C.data(), N, N, N, threads); });
    csv.row("matmul", type_name<T>(), N, B::name, threads, ms, "GFLOP/s", 2.0 * N * N * N / (ms * 1e6));
}

template <class T>
void run_type(Csv& csv, const std::vector<std::size_t>& thread_list) {
    sanity_check<T>();
    for (std::size_t n : {std::size_t{1} << 10, std::size_t{1} << 14, std::size_t{1} << 18, std::size_t{1} << 22}) {
        auto a = make_data<T>(n, 11), b = make_data<T>(n, 12);
        std::vector<T> out(n);
        for (std::size_t th : thread_list) {
            bench_backend<hw::Scalar>(csv, th, n, a, b, out);
            bench_backend<hw::Simd>(csv, th, n, a, b, out);
        }
    }
    for (std::size_t N : {64, 128, 256, 512}) {
        auto A = make_data<T>(N * N, 21), Bm = make_data<T>(N * N, 22);
        std::vector<T> C(N * N);
        for (std::size_t th : thread_list) {
            bench_matmul<hw::Scalar>(csv, th, N, A, Bm, C);
            bench_matmul<hw::Simd>(csv, th, N, A, Bm, C);
        }
    }
}

int main() {
    std::vector<std::size_t> thread_list{1};
    std::size_t maxt = hw::ThreadPool::instance().max_threads();
    if (maxt > 1) thread_list.push_back(maxt);
    std::fprintf(stderr, "simd bytes: %zu, max threads: %zu\n", hw::kSimdBytes, maxt);
    std::printf("op,type,size,backend,threads,time_ms,metric,value\n");
    Csv csv;
    run_type<std::int32_t>(csv, thread_list);
    run_type<float>(csv, thread_list);
    run_type<double>(csv, thread_list);
    std::fprintf(stderr, "sanity checks passed\n");
}
