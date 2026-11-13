// Тесты на корректность
// Запуск: ./hw_tests

// Проверяются:
// 1. размеры вокруг границ вектора и границ параллельных кусков (хвосты), невыровненные указатели
// 2. отсутствие записи за пределы выходного массива
// 3. разное число потоков (по умолчанию тесты принудительно поднимают пул на 4 потока даже на 1 ядре)
// 4. int32 с переполнением (по модулю 2^32), float/double с допусками
// 5. SiLU/GELU против эталона в long double, точность печатается в сводке
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../include/activations.hpp"


static int g_checks = 0, g_failures = 0;

static void report_fail(const char* file, int line, const std::string& msg) {
    ++g_failures;
    if (g_failures <= 40) std::fprintf(stderr, "FAIL %s:%d  %s\n", file, line, msg.c_str());
}
#define CHECK(cond, msg)                                       \
    do {                                                       \
        ++g_checks;                                            \
        if (!(cond)) report_fail(__FILE__, __LINE__, (msg));   \
    } while (0)

static std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

template <class T>
const char* tname() {
    if constexpr (std::is_same_v<T, std::int32_t>) return "int32";
    else if constexpr (std::is_same_v<T, float>) return "float32";
    else return "float64";
}


enum class Mode { Wide, NonZero, IntValued, Unit };

// Wide: int32 на всём диапазоне (проверка переполнения), float [-100, 100]
// NonZero: без нулей (для деления), ints в [-1000, 1000], floats |x| в [0.5, 100]
// IntValued: небольшие целые, записанные в T (суммы точны при любом порядке)
// Unit: float/double в [-1, 1]
template <class T>
std::vector<T> make(std::size_t n, std::uint32_t seed, Mode mode) {
    std::mt19937 rng(seed);
    std::vector<T> v(n);
    for (auto& x : v) {
        if constexpr (std::is_same_v<T, std::int32_t>) {
            switch (mode) {
                case Mode::Wide: x = static_cast<std::int32_t>(rng()); break;
                case Mode::NonZero: {
                    std::int32_t m = static_cast<std::int32_t>(rng() % 1000) + 1;
                    x = (rng() & 1) ? m : -m;
                    break;
                }
                default: x = static_cast<std::int32_t>(rng() % 17) - 8; break;
            }
        } else {
            std::uniform_real_distribution<double> u(0.0, 1.0);
            switch (mode) {
                case Mode::Wide: x = static_cast<T>(u(rng) * 200.0 - 100.0); break;
                case Mode::NonZero: {
                    double m = 0.5 + u(rng) * 99.5;
                    x = static_cast<T>((rng() & 1) ? m : -m);
                    break;
                }
                case Mode::Unit: x = static_cast<T>(u(rng) * 2.0 - 1.0); break;
                default: x = static_cast<T>(static_cast<int>(rng() % 17) - 8); break;
            }
        }
    }
    return v;
}

template <class T>
T wrap_add(T a, T b) {
    if constexpr (std::is_same_v<T, std::int32_t>)
        return static_cast<T>(static_cast<std::uint32_t>(a) + static_cast<std::uint32_t>(b));
    else return a + b;
}
template <class T>
T wrap_sub(T a, T b) {
    if constexpr (std::is_same_v<T, std::int32_t>)
        return static_cast<T>(static_cast<std::uint32_t>(a) - static_cast<std::uint32_t>(b));
    else return a - b;
}
template <class T>
T wrap_mul(T a, T b) {
    if constexpr (std::is_same_v<T, std::int32_t>)
        return static_cast<T>(static_cast<std::uint32_t>(a) * static_cast<std::uint32_t>(b));
    else return a * b;
}

static const std::vector<std::size_t> kSizes = {0,   1,   2,    3,    4,     5,     7,     8,     9,     15,
                                                16,  17,  31,   32,   33,    63,    64,    65,    127,   128,
                                                129, 255, 1000, 4097, 65535, 65536, 65537, 131073, 200003, 300007};

static std::vector<std::size_t> threads_for(std::size_t n) {
    if (n <= 4097) return {0, 1};
    return {0, 1, 2, 3, 5};
}

// поэлементные операции
template <class B, class T>
void test_binary_ops() {
    struct Case {
        const char* name;
        std::function<void(const T*, const T*, T*, std::size_t, std::size_t)> run;
        std::function<T(T, T)> ref;
        Mode mode_b;
    };
    std::vector<Case> cases = {
        {"add", [](const T* a, const T* b, T* o, std::size_t n, std::size_t th) { hw::add<B>(a, b, o, n, th); },
         [](T x, T y) { return wrap_add(x, y); }, Mode::Wide},
        {"sub", [](const T* a, const T* b, T* o, std::size_t n, std::size_t th) { hw::sub<B>(a, b, o, n, th); },
         [](T x, T y) { return wrap_sub(x, y); }, Mode::Wide},
        {"mul", [](const T* a, const T* b, T* o, std::size_t n, std::size_t th) { hw::mul<B>(a, b, o, n, th); },
         [](T x, T y) { return wrap_mul(x, y); }, Mode::Wide},
        {"div", [](const T* a, const T* b, T* o, std::size_t n, std::size_t th) { hw::div<B>(a, b, o, n, th); },
         [](T x, T y) { return static_cast<T>(x / y); }, Mode::NonZero},
    };
    const T sentinel = std::is_same_v<T, std::int32_t> ? static_cast<T>(0x5A5A5A5A) : static_cast<T>(12345.5);
    for (auto& c : cases) {
        for (std::size_t n : kSizes) {
            for (std::size_t off : {std::size_t{0}, std::size_t{1}, std::size_t{3}}) {
                auto a = make<T>(n + off, 100 + n, c.mode_b == Mode::NonZero ? Mode::NonZero : Mode::Wide);
                auto b = make<T>(n + off, 200 + n, c.mode_b);
                for (std::size_t th : threads_for(n)) {
                    std::vector<T> out(n + off + 16, sentinel);
                    c.run(a.data() + off, b.data() + off, out.data() + off, n, th);
                    bool ok = true;
                    for (std::size_t i = 0; i < n && ok; ++i) {
                        T want = c.ref(a[off + i], b[off + i]);
                        if (!(out[off + i] == want)) {
                            report_fail(__FILE__, __LINE__,
                                        fmt("%s %s %s n=%zu off=%zu th=%zu i=%zu got=%g want=%g", B::name, tname<T>(),
                                            c.name, n, off, th, i, double(out[off + i]), double(want)));
                            ok = false;
                        }
                    }
                    ++g_checks;
                    bool guard = true;  // за пределы [off, off+n) ничего не записано
                    for (std::size_t i = 0; i < off; ++i) guard &= (out[i] == sentinel);
                    for (std::size_t i = off + n; i < out.size(); ++i) guard &= (out[i] == sentinel);
                    CHECK(guard, fmt("%s %s %s n=%zu off=%zu th=%zu wrote outside output", B::name, tname<T>(), c.name,
                                     n, off, th));
                    if (!ok) return;
                }
            }
        }
    }
}

// проверка, что transform работает с произвольной операцией
struct MulAddA {
    template <class X>
    X operator()(const X& a, const X& b) const { return a * b + a; }
};

template <class B, class T>
void test_custom_functor() {
    for (std::size_t n : {std::size_t{0}, std::size_t{5}, std::size_t{37}, std::size_t{100003}}) {
        auto a = make<T>(n, 1, Mode::IntValued), b = make<T>(n, 2, Mode::IntValued);
        std::vector<T> out(n);
        hw::transform<B>(a.data(), b.data(), out.data(), n, MulAddA{}, 0);
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) ok &= (out[i] == wrap_add(wrap_mul(a[i], b[i]), a[i]));
        CHECK(ok, fmt("%s %s custom functor n=%zu", B::name, tname<T>(), n));
    }
}

// reduction
template <class T>
double sum_ref_exact(const std::vector<T>& v) {
    if constexpr (std::is_same_v<T, std::int32_t>) {
        std::uint32_t s = 0;
        for (T x : v) s += static_cast<std::uint32_t>(x);
        return static_cast<double>(static_cast<std::int32_t>(s));
    } else {
        long double s = 0;
        for (T x : v) s += x;
        return static_cast<double>(s);
    }
}

template <class B, class T>
void test_reductions() {
    for (std::size_t n : kSizes) {
        for (Mode mode : {Mode::IntValued, Mode::Wide}) {
            // для float и double Wide проверяется ниже с допуском; точное сравнение только для int32 и IntValued
            const bool exact = std::is_same_v<T, std::int32_t> || mode == Mode::IntValued;
            auto v = make<T>(n, 7 + n, mode);
            double ref = sum_ref_exact(v);
            for (std::size_t th : threads_for(n)) {
                T got = hw::sum<B>(v.data(), n, th);
                double got_mean = hw::mean<B>(v.data(), n, th);
                double want_mean = n ? ref / static_cast<double>(n) : 0.0;
                if (exact) {
                    CHECK(static_cast<double>(got) == ref,
                          fmt("%s %s sum n=%zu th=%zu got=%.17g want=%.17g", B::name, tname<T>(), n, th, double(got), ref));
                    CHECK(got_mean == want_mean, fmt("%s %s mean n=%zu th=%zu", B::name, tname<T>(), n, th));
                } else {
                    double eps = std::numeric_limits<T>::epsilon();
                    double tol = 0.5 * eps * static_cast<double>(n + 1) * 100.0;
                    CHECK(std::fabs(static_cast<double>(got) - ref) <= tol,
                          fmt("%s %s sum(wide) n=%zu th=%zu got=%.9g want=%.9g tol=%.3g", B::name, tname<T>(), n, th,
                              double(got), ref, tol));
                    CHECK(std::fabs(got_mean - want_mean) <= tol / std::max<double>(1.0, double(n)) + 1e-300,
                          fmt("%s %s mean(wide) n=%zu th=%zu", B::name, tname<T>(), n, th));
                }
            }
        }
    }
    CHECK(hw::sum<B>(static_cast<const T*>(nullptr), 0, 0) == T{0}, "sum of empty");
    CHECK(hw::mean<B>(static_cast<const T*>(nullptr), 0, 0) == 0.0, "mean of empty");

    // reduce с другой операцией: произведение int32 по модулю 2^32 (порядок не важен)
    if constexpr (std::is_same_v<T, std::int32_t>) {
        for (std::size_t n : {std::size_t{1}, std::size_t{33}, std::size_t{70001}, std::size_t{300007}}) {
            auto v = make<T>(n, 5, Mode::Wide);
            std::uint32_t p = 1;
            for (T x : v) p *= static_cast<std::uint32_t>(x);
            for (std::size_t th : threads_for(n))
                CHECK(hw::reduce<B>(v.data(), n, T{1}, hw::Mul{}, th) == static_cast<T>(p),
                      fmt("%s int32 product n=%zu th=%zu", B::name, n, th));
        }
    }
}

// matmul
struct Shape {
    std::size_t M, K, N;
};

template <class B, class T>
void test_matmul() {
    const std::vector<Shape> shapes = {{1, 1, 1},   {1, 1, 5},   {5, 1, 1},     {1, 7, 1},      {3, 4, 5},
                                       {4, 4, 4},   {4, 16, 16}, {5, 17, 33},    {8, 8, 8},      {16, 16, 16},
                                       {31, 32, 33}, {67, 53, 71}, {9, 130, 260}, {130, 129, 131}, {257, 3, 5},
                                       {200, 200, 200}, {3, 0, 4}};
    for (auto s : shapes) {
        auto A = make<T>(s.M * s.K, 11 + s.M, Mode::Wide), Bm = make<T>(s.K * s.N, 13 + s.N, Mode::Wide);
        for (std::size_t th : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
            std::vector<T> C(s.M * s.N, T(-7));
            hw::matmul<B>(A.data(), Bm.data(), C.data(), s.M, s.K, s.N, th);
            bool ok = true;
            for (std::size_t i = 0; i < s.M && ok; ++i)
                for (std::size_t j = 0; j < s.N && ok; ++j) {
                    if constexpr (std::is_same_v<T, std::int32_t>) {
                        std::uint32_t acc = 0;
                        for (std::size_t k = 0; k < s.K; ++k)
                            acc += static_cast<std::uint32_t>(A[i * s.K + k]) * static_cast<std::uint32_t>(Bm[k * s.N + j]);
                        if (C[i * s.N + j] != static_cast<T>(acc)) ok = false;
                    } else {
                        long double acc = 0, abs_acc = 0;
                        for (std::size_t k = 0; k < s.K; ++k) {
                            long double p = static_cast<long double>(A[i * s.K + k]) * Bm[k * s.N + j];
                            acc += p;
                            abs_acc += std::fabs(p);
                        }
                        // стандартная оценка ошибки скалярного произведения: K * eps * sum|a_k b_k|
                        long double tol = static_cast<long double>(s.K) * std::numeric_limits<T>::epsilon() * abs_acc;
                        if (std::fabs(static_cast<long double>(C[i * s.N + j]) - acc) > tol) ok = false;
                    }
                }
            CHECK(ok, fmt("%s %s matmul %zux%zux%zu th=%zu", B::name, tname<T>(), s.M, s.K, s.N, th));
        }
    }
}

// activations
struct AccStat {
    double max_rel = 0, max_abs = 0;
};

static long double silu_ref(long double x) { return x / (1.0L + std::exp(-x)); }
static long double gelu_y(long double x) { return 0.7978845608028653558798921L * (x + 0.044715L * x * x * x); }
static long double gelu_ref(long double x) { return x / (1.0L + std::exp(-2.0L * gelu_y(x))); }
static long double gelu_tanh_form(long double x) { return 0.5L * x * (1.0L + std::tanh(gelu_y(x))); }

struct Relu {  // проверка, что unary_transform принимает произвольный функтор
    template <class R>
    R operator()(R x) const { return vmax(x, R::broadcast(typename R::value_type(0))); }
};

template <class B, class T>
AccStat test_activation(const char* fname, bool is_silu, double rtol, double atol) {
    using O = hw::ActOut<T>;
    std::vector<T> x;
    if constexpr (std::is_same_v<T, std::int32_t>) {
        for (int i = -300; i <= 300; ++i) x.push_back(i);
        for (std::int32_t v : {INT_MAX, INT_MIN, INT_MAX - 1, INT_MIN + 1, 1000000, -1000000, 100000, -100000})
            x.push_back(v);
        std::mt19937 rng(3);
        for (int i = 0; i < 20000; ++i) x.push_back(static_cast<std::int32_t>(rng()) / (1 + (rng() % 1000000)));
    } else {
        for (int i = -10000; i <= 10000; ++i) x.push_back(static_cast<T>(i * 0.003));
        std::mt19937 rng(9);
        std::uniform_real_distribution<double> u(-100, 100);
        for (int i = 0; i < 20000; ++i) x.push_back(static_cast<T>(u(rng)));
        const T mx = std::numeric_limits<T>::max();
        for (double v : {0.0, 1e-30, 1e-8, 87.0, 88.0, 89.0, 100.0, 710.0, 800.0, 1e4, 1e20})
            for (double sgn : {1.0, -1.0}) x.push_back(static_cast<T>(sgn * v));
        x.push_back(mx);
        x.push_back(-mx);
    }
    std::vector<T> xs = x;
    xs.push_back(T(0.5));
    xs.push_back(T(-0.5));
    xs.push_back(T(2));
    const std::size_t n = xs.size();
    std::vector<O> out(n, O(-1));
    for (std::size_t th : {std::size_t{0}, std::size_t{1}}) {
        if (is_silu) hw::silu<B>(xs.data(), out.data(), n, th);
        else hw::gelu<B>(xs.data(), out.data(), n, th);
    }
    AccStat st;
    const double tiny = std::is_same_v<O, float> ? 1e-30 : 1e-290;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        long double xi = static_cast<long double>(static_cast<O>(xs[i]));  // вход после приведения к O
        long double ref = is_silu ? silu_ref(xi) : gelu_ref(xi);
        double got = static_cast<double>(out[i]);
        double err = std::fabs(static_cast<double>(got - ref));
        if (err > atol + rtol * std::fabs(static_cast<double>(ref))) {
            if (++bad <= 3)
                std::fprintf(stderr, "FAIL %s %s %s x=%.9g got=%.9g ref=%.9g\n", B::name, tname<T>(), fname, double(xi),
                             got, double(ref));
        }
        st.max_abs = std::max(st.max_abs, err);
        if (std::fabs(static_cast<double>(ref)) > tiny) st.max_rel = std::max(st.max_rel, err / std::fabs(static_cast<double>(ref)));
    }
    CHECK(bad == 0, fmt("%s %s %s: %zu values out of tolerance", B::name, tname<T>(), fname, bad));
    return st;
}

template <class B, class T>
void test_unary_custom() {
    using O = hw::ActOut<T>;
    std::vector<T> x = make<T>(100003, 4, Mode::Wide);
    std::vector<O> out(x.size());
    hw::unary_transform<B>(x.data(), out.data(), x.size(), Relu{}, 0);
    bool ok = true;
    for (std::size_t i = 0; i < x.size(); ++i) ok &= (out[i] == std::max<O>(static_cast<O>(x[i]), O(0)));
    CHECK(ok, fmt("%s %s relu via unary_transform", B::name, tname<T>()));
}

template <class B, class T>
void test_activation_special() {
    if constexpr (std::is_floating_point_v<T>) {
        const T inf = std::numeric_limits<T>::infinity();
        T in[4] = {inf, -inf, T(0), T(-0.0)}, o[4];
        for (int which = 0; which < 2; ++which) {
            if (which == 0) hw::silu<B>(in, o, 4, 1);
            else hw::gelu<B>(in, o, 4, 1);
            CHECK(o[0] == inf, fmt("%s %s %s(+inf)", B::name, tname<T>(), which ? "gelu" : "silu"));
            CHECK(std::fabs(o[1]) < T(1e-20), fmt("%s %s %s(-inf)", B::name, tname<T>(), which ? "gelu" : "silu"));
            CHECK(o[2] == T(0) && o[3] == T(0), fmt("%s %s %s(0)", B::name, tname<T>(), which ? "gelu" : "silu"));
        }
    }
}

static void test_gelu_vs_exact() {
    for (double x = -3; x <= 10; x += 0.01) {
        double a = static_cast<double>(gelu_ref(x)), b = static_cast<double>(gelu_tanh_form(x));
        CHECK(std::fabs(a - b) <= 1e-15 * std::max(1.0, std::fabs(b)), fmt("gelu identity x=%g", x));
    }
    double worst = 0;
    for (double x = -10; x <= 10; x += 0.001) {
        double exact = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        worst = std::max(worst, std::fabs(static_cast<double>(gelu_ref(x)) - exact));
    }
    std::printf("INFO  max |gelu_tanh - gelu_erf| on [-10,10] = %.3e\n", worst);
    CHECK(worst < 1e-3, "tanh approximation of GELU too far from erf GELU");
}

// threads
// много параллельных регионов подряд и одновременные вызовы из разных потоков (пул обязан сериализовать регионы)
template <class B>
void test_pool_stress() {
    std::mt19937 rng(77);
    for (int it = 0; it < 300; ++it) {
        std::size_t n = 65536 + rng() % 400000;
        std::size_t th = rng() % 6;
        auto v = make<std::int32_t>(n, it, Mode::IntValued);
        std::int64_t want = 0;
        for (auto x : v) want += x;
        CHECK(hw::sum<B>(v.data(), n, th) == static_cast<std::int32_t>(want), fmt("pool stress it=%d n=%zu th=%zu", it, n, th));
    }
    std::atomic<int> bad{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t)
        ts.emplace_back([&, t] {
            auto v = make<float>(250000 + t, 31 + t, Mode::IntValued);
            double want = sum_ref_exact(v);
            for (int it = 0; it < 40; ++it)
                if (static_cast<double>(hw::sum<B>(v.data(), v.size(), 0)) != want) ++bad;
        });
    for (auto& t : ts) t.join();
    CHECK(bad == 0, "concurrent callers got wrong sums");
}

template <class B>
void run_backend() {
    std::printf("== backend: %s ==\n", B::name);
    test_binary_ops<B, std::int32_t>();
    test_binary_ops<B, float>();
    test_binary_ops<B, double>();
    test_custom_functor<B, std::int32_t>();
    test_custom_functor<B, float>();
    test_custom_functor<B, double>();
    test_reductions<B, std::int32_t>();
    test_reductions<B, float>();
    test_reductions<B, double>();
    test_matmul<B, std::int32_t>();
    test_matmul<B, float>();
    test_matmul<B, double>();

    auto show = [&](const char* f, const char* t, AccStat s) {
        std::printf("ACCURACY %-6s %-5s %-8s max_rel=%.3e max_abs=%.3e\n", B::name, f, t, s.max_rel, s.max_abs);
    };
    // допуски: float -- несколько единиц в последнем знаке, float64 -- ~1e-13
    show("silu", "float32", test_activation<B, float>("silu", true, 1e-6, 1e-30));
    show("gelu", "float32", test_activation<B, float>("gelu", false, 3e-5, 1e-30));
    show("silu", "float64", test_activation<B, double>("silu", true, 1e-14, 1e-290));
    show("gelu", "float64", test_activation<B, double>("gelu", false, 1e-12, 1e-290));
    show("silu", "int32", test_activation<B, std::int32_t>("silu", true, 1e-6, 1e-30));
    show("gelu", "int32", test_activation<B, std::int32_t>("gelu", false, 3e-5, 1e-30));
    test_activation_special<B, float>();
    test_activation_special<B, double>();
    test_unary_custom<B, std::int32_t>();
    test_unary_custom<B, float>();
    test_unary_custom<B, double>();
    test_pool_stress<B>();
}

int main() {
#ifndef _WIN32
    setenv("HW_THREADS", "4", 0);
#endif
    std::printf("simd bytes: %zu, pool threads: %zu\n", hw::kSimdBytes, hw::ThreadPool::instance().max_threads());
    run_backend<hw::Scalar>();
    run_backend<hw::Simd>();
    test_gelu_vs_exact();

    {
        auto x = make<float>(100003, 8, Mode::Wide);
        std::vector<float> a(x.size()), b(x.size());
        hw::gelu<hw::Scalar>(x.data(), a.data(), x.size(), 1);
        hw::gelu<hw::Simd>(x.data(), b.data(), x.size(), 0);
        double worst = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            worst = std::max(worst, std::fabs(double(a[i]) - double(b[i])) / std::max(1e-30, std::fabs(double(a[i]))));
        std::printf("INFO  max rel diff Scalar vs Simd gelu float32 = %.3e\n", worst);
        CHECK(worst < 1e-6, "Scalar and Simd activations diverge");
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
