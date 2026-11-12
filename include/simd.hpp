// Единый интерфейс "регистра" для Scalar и SIMD реализаций (обобщённый подход)
// Backend выбирает, чем будет Reg: ScalarReg<T> (1 элемент) или Pack<T> (вектор)
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


template <class T>
struct ScalarReg {
    static_assert(is_supported_v<T>, "supported types: int32_t, float, double");
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
};


// реализация на векторных расширениях GCC/Clang (__attribute__((vector_size)))
template <class T>
struct Pack {
    static_assert(is_supported_v<T>, "supported types: int32_t, float, double");
    static constexpr std::size_t width = kSimdBytes / sizeof(T);
    typedef T vec_t __attribute__((vector_size(kSimdBytes)));
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

    template <class Op>
    T hreduce(Op op) const {
        T tmp[width];
        store(tmp);
        T r = tmp[0];
        for (std::size_t i = 1; i < width; ++i) r = op(r, tmp[i]);
        return r;
    }
};


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
