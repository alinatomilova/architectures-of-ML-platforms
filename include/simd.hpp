// Единый интерфейс "регистра" для Scalar и SIMD реализаций
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hw {

template <class T>
inline constexpr bool is_supported_v =
    std::is_same_v<T, std::int32_t> || std::is_same_v<T, float> || std::is_same_v<T, double>;

// ширина SIMD-регистра в байтах выбирается по флагам компиляции (-march=native)
#if defined(__AVX512F__)
inline constexpr std::size_t kSimdBytes = 64;
#elif defined(__AVX2__) || defined(__AVX__)
inline constexpr std::size_t kSimdBytes = 32;
#else
inline constexpr std::size_t kSimdBytes = 16;  // SSE2 / NEON
#endif

// Scalar
template <class T>
struct ScalarReg {
    static_assert(is_supported_v<T>, "supported types: int32_t, float, double");
    using value_type = T;
    static constexpr std::size_t width = 1;
    T v;

    static ScalarReg load(const T* p) { return {*p}; }
    static ScalarReg broadcast(T x) { return {x}; }
    void store(T* p) const { *p = v; }

    friend ScalarReg operator+(ScalarReg a, ScalarReg b) { return {a.v + b.v}; }
    friend ScalarReg operator-(ScalarReg a, ScalarReg b) { return {a.v - b.v}; }
    friend ScalarReg operator*(ScalarReg a, ScalarReg b) { return {a.v * b.v}; }
    friend ScalarReg operator/(ScalarReg a, ScalarReg b) { return {a.v / b.v}; }

    template <class Op>
    T hreduce(Op) const { return v; }

    friend ScalarReg vmin(ScalarReg a, ScalarReg b) { return {b.v < a.v ? b.v : a.v}; }
    friend ScalarReg vmax(ScalarReg a, ScalarReg b) { return {b.v > a.v ? b.v : a.v}; }


    ScalarReg scale_pow2(ScalarReg n) const {
        static_assert(std::is_floating_point_v<T>);
        using I = std::conditional_t<sizeof(T) == 4, std::int32_t, std::int64_t>;
        constexpr int mant = sizeof(T) == 4 ? 23 : 52;
        constexpr I bias = sizeof(T) == 4 ? 127 : 1023;
        I bits = (static_cast<I>(n.v) + bias) << mant;
        T p;
        std::memcpy(&p, &bits, sizeof p);
        return {v * p};
    }
};

template <class To, class From>
ScalarReg<To> reg_cast(ScalarReg<From> a) { return {static_cast<To>(a.v)}; }


// SIMD 
// Реализация на векторных расширениях GCC/Clang (__attribute__((vector_size)))
// Компилятор сам отображает +, -, *, / на AVX/AVX2/AVX-512/NEON инструкции
// Для int32 деление векторизуется компилятором как набор скалярных делений (в AVX2 нет целочисленного деления)
template <class T>
struct Pack {
    static_assert(is_supported_v<T>, "supported types: int32_t, float, double");
    using value_type = T;
    static constexpr std::size_t width = kSimdBytes / sizeof(T);
    typedef T vec_t __attribute__((vector_size(kSimdBytes)));
    using int_t = std::conditional_t<sizeof(T) == 4, std::int32_t, std::int64_t>;
    typedef int_t ivec_t __attribute__((vector_size(kSimdBytes)));
    vec_t v;

    static Pack load(const T* p) {  // невыровненная загрузка
        Pack r;
        std::memcpy(&r.v, p, sizeof(vec_t));
        return r;
    }
    static Pack broadcast(T x) {
        Pack r;
        r.v = vec_t{} + x;
        return r;
    }
    void store(T* p) const { std::memcpy(p, &v, sizeof(vec_t)); }

    friend Pack operator+(Pack a, Pack b) { return {a.v + b.v}; }
    friend Pack operator-(Pack a, Pack b) { return {a.v - b.v}; }
    friend Pack operator*(Pack a, Pack b) { return {a.v * b.v}; }
    friend Pack operator/(Pack a, Pack b) { return {a.v / b.v}; }

    friend Pack vmin(Pack a, Pack b) { return {b.v < a.v ? b.v : a.v}; }
    friend Pack vmax(Pack a, Pack b) { return {b.v > a.v ? b.v : a.v}; }

    Pack scale_pow2(Pack n) const {
        static_assert(std::is_floating_point_v<T>);
        constexpr int mant = sizeof(T) == 4 ? 23 : 52;
        constexpr int_t bias = sizeof(T) == 4 ? 127 : 1023;
        ivec_t in = __builtin_convertvector(n.v, ivec_t);
        ivec_t bits = (in + bias) << mant;
        vec_t p;
        std::memcpy(&p, &bits, sizeof p);
        return {v * p};
    }

    // горизонтальная свёртка регистра произвольной бинарной операцией
    template <class Op>
    T hreduce(Op op) const {
        T tmp[width];
        store(tmp);
        T r = tmp[0];
        for (std::size_t i = 1; i < width; ++i) r = op(r, tmp[i]);
        return r;
    }
};

// поэлементное приведение типа (ширина регистров должна совпадать: int32 <-> float, double <-> int64)
template <class To, class From>
Pack<To> reg_cast(Pack<From> a) {
    if constexpr (std::is_same_v<To, From>) {
        return a;
    } else {
        static_assert(Pack<To>::width == Pack<From>::width, "reg_cast needs equal lane counts");
        return {__builtin_convertvector(a.v, typename Pack<To>::vec_t)};
    }
}

// Backends
struct Scalar {
    static constexpr const char* name = "scalar";
    template <class T>
    using Reg = ScalarReg<T>;
};

struct Simd {
    static constexpr const char* name = "simd";
    template <class T>
    using Reg = Pack<T>;
};

}  // namespace hw
