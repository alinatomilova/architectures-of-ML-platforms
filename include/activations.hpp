// Функции активации SiLU и GELU (аппроксимации)

//   SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//   GELU(x) = x * Φ(x) ~ 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
//           = x / (1 + exp(-2 sqrt(2/pi) (x + 0.044715 x^3)))      (тождество 0.5(1+tanh(y)) = sigmoid(2y))

// exp считается собственной аппроксимацией (нет зависимости от libm / SVML):
// exp(x) = 2^n * exp(r),  n = round(x / ln 2),  r = x - n ln 2,  exp(r) - полином на [-ln2/2, ln2/2]
#pragma once
 
#include <cstddef>
#include <cstdint>
#include <type_traits>
 
#include "ops.hpp"
 
namespace hw {
 
template <class T>
using ActOut = std::conditional_t<std::is_same_v<T, std::int32_t>, float, T>;
 
namespace math {
 
// приближённая exp(x) для Reg<float> / Reg<double>
// относительная ошибка ~1e-7 (float), ~1e-15 (double)
template <class R>
R exp(R x) {
    using T = typename R::value_type;
    static_assert(std::is_floating_point_v<T>, "exp: float or double only");
    auto c = [](double d) { return R::broadcast(static_cast<T>(d)); };
 
    if constexpr (std::is_same_v<T, float>) {
        x = vmin(vmax(x, c(-87.0)), c(88.0));
        // n = round(x * log2(e))
        R magic = c(12582912.0);
        R n = (x * c(1.44269504088896341) + magic) - magic;
        // r = x - n ln2
        R r = (x - n * c(0.693359375)) - n * c(-2.12194440e-4);
        // exp(r) ~ 1 + r + r^2 * P(r)
        R p = c(1.9875691500e-4);
        p = p * r + c(1.3981999507e-3);
        p = p * r + c(8.3334519073e-3);
        p = p * r + c(4.1665795894e-2);
        p = p * r + c(1.6666665459e-1);
        p = p * r + c(5.0000001201e-1);
        R y = p * (r * r) + r + c(1.0);
        return y.scale_pow2(n);
    } else {
        x = vmin(vmax(x, c(-700.0)), c(700.0));
        R magic = c(6755399441055744.0);  // 1.5 * 2^52
        R n = (x * c(1.4426950408889634074) + magic) - magic;
        R r = (x - n * c(6.93147180369123816490e-01)) - n * c(1.90821492927058770002e-10);
        // Ряд Тейлора до r^13 (|r| <= 0.3466: остаток < 1e-17), схема Горнера
        R p = c(1.0 / 6227020800.0);  // 1/13!
        p = p * r + c(1.0 / 479001600.0);
        p = p * r + c(1.0 / 39916800.0);
        p = p * r + c(1.0 / 3628800.0);
        p = p * r + c(1.0 / 362880.0);
        p = p * r + c(1.0 / 40320.0);
        p = p * r + c(1.0 / 5040.0);
        p = p * r + c(1.0 / 720.0);
        p = p * r + c(1.0 / 120.0);
        p = p * r + c(1.0 / 24.0);
        p = p * r + c(1.0 / 6.0);
        p = p * r + c(0.5);
        p = p * r + c(1.0);
        p = p * r + c(1.0);
        return p.scale_pow2(n);
    }
}
 
// x / (1 + exp(-s)): общая часть SiLU (s = x) и GELU (s = 2 sqrt(2/pi) (x + 0.044715 x^3)).
template <class R>
R x_over_one_plus_exp_neg(R x, R s) {
    using T = typename R::value_type;
    constexpr T lo = std::is_same_v<T, float> ? T(-87) : T(-700);
    R zero = R::broadcast(T(0)), one = R::broadcast(T(1));
    return vmax(x, R::broadcast(lo)) / (one + exp(zero - s));
}
 
}  // namespace math
 

struct Silu {
    template <class R>
    R operator()(R x) const {
        return math::x_over_one_plus_exp_neg(x, x);
    }
};
 
struct Gelu {
    template <class R>
    R operator()(R x) const {
        using T = typename R::value_type;
        R x3 = x * x * x;
        // s = 2 * sqrt(2/pi) * (x + 0.044715 x^3)
        R s = R::broadcast(T(1.5957691216057308)) * (x + R::broadcast(T(0.044715)) * x3);
        return math::x_over_one_plus_exp_neg(x, s);
    }
};
 
inline constexpr std::size_t kActGrain = std::size_t{1} << 14;
 
namespace detail {
 
template <class B, class In, class Out, class F>
void unary_range(const In* in, Out* out, std::size_t n, F f) {
    using RI = typename B::template Reg<In>;
    using RO = typename B::template Reg<Out>;
    static_assert(RI::width == RO::width, "input and output lanes must match");
    constexpr std::size_t W = RI::width;
    std::size_t i = 0;
    for (; i + W <= n; i += W) f(reg_cast<Out>(RI::load(in + i))).store(out + i);
    for (; i < n; ++i) f(reg_cast<Out>(ScalarReg<In>::load(in + i))).store(out + i);
}
 
}  // namespace detail
 
// out[i] = f(in[i])
template <class B, class T, class F>
void unary_transform(const T* in, ActOut<T>* out, std::size_t n, F f, std::size_t threads = 0) {
    parallel_for(0, n, kActGrain, threads, [&](std::size_t s, std::size_t e) {
        detail::unary_range<B, T, ActOut<T>>(in + s, out + s, e - s, f);
    });
}
 
template <class B, class T>
void silu(const T* in, ActOut<T>* out, std::size_t n, std::size_t threads = 0) {
    unary_transform<B>(in, out, n, Silu{}, threads);
}
 
template <class B, class T>
void gelu(const T* in, ActOut<T>* out, std::size_t n, std::size_t threads = 0) {
    unary_transform<B>(in, out, n, Gelu{}, threads);
}
 
}  // namespace hw
 
