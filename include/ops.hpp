// Обобщённые алгоритмы
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "parallel.hpp"
#include "simd.hpp"

namespace hw {

struct Add {
    template <class X> X operator()(const X& a, const X& b) const { return a + b; }
};
struct Sub {
    template <class X> X operator()(const X& a, const X& b) const { return a - b; }
};
struct Mul {
    template <class X> X operator()(const X& a, const X& b) const { return a * b; }
};
struct Div {
    template <class X> X operator()(const X& a, const X& b) const { return a / b; }
};

// min размер куска работы на поток (в элементах)
inline constexpr std::size_t kElemGrain = std::size_t{1} << 16;


namespace detail {

template <class B, class T, class Op>
void binary_range(const T* a, const T* b, T* out, std::size_t n, Op op) {
    using R = typename B::template Reg<T>;
    constexpr std::size_t W = R::width;
    std::size_t i = 0;
    for (; i + W <= n; i += W) op(R::load(a + i), R::load(b + i)).store(out + i);
    for (; i < n; ++i) out[i] = op(a[i], b[i]);
}

template <class B, class T, class Op>
T reduce_range(const T* a, std::size_t n, T identity, Op op) {
    using R = typename B::template Reg<T>;
    constexpr std::size_t W = R::width;
    // 4 независимых аккумулятора скрывают латентность операции
    R acc0 = R::broadcast(identity), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 4 * W <= n; i += 4 * W) {
        acc0 = op(acc0, R::load(a + i));
        acc1 = op(acc1, R::load(a + i + W));
        acc2 = op(acc2, R::load(a + i + 2 * W));
        acc3 = op(acc3, R::load(a + i + 3 * W));
    }
    for (; i + W <= n; i += W) acc0 = op(acc0, R::load(a + i));
    T r = op(op(acc0, acc1), op(acc2, acc3)).hreduce(op);
    for (; i < n; ++i) r = op(r, a[i]);
    return r;
}

}  // namespace detail


template <class B, class T, class Op>
void transform(const T* a, const T* b, T* out, std::size_t n, Op op, std::size_t threads = 0) {
    parallel_for(0, n, kElemGrain, threads, [&](std::size_t s, std::size_t e) {
        detail::binary_range<B>(a + s, b + s, out + s, e - s, op);
    });
}

template <class B, class T>
void add(const T* a, const T* b, T* out, std::size_t n, std::size_t threads = 0) {
    transform<B>(a, b, out, n, Add{}, threads);
}
template <class B, class T>
void sub(const T* a, const T* b, T* out, std::size_t n, std::size_t threads = 0) {
    transform<B>(a, b, out, n, Sub{}, threads);
}
template <class B, class T>
void mul(const T* a, const T* b, T* out, std::size_t n, std::size_t threads = 0) {
    transform<B>(a, b, out, n, Mul{}, threads);
}

template <class B, class T>
void div(const T* a, const T* b, T* out, std::size_t n, std::size_t threads = 0) {
    transform<B>(a, b, out, n, Div{}, threads);
}


template <class B, class T, class Op>
T reduce(const T* a, std::size_t n, T identity, Op op, std::size_t threads = 0) {
    return parallel_reduce<T>(
        0, n, kElemGrain, threads, identity,
        [&](std::size_t s, std::size_t e) { return detail::reduce_range<B>(a + s, e - s, identity, op); }, op);
}

template <class B, class T>
T sum(const T* a, std::size_t n, std::size_t threads = 0) {
    return reduce<B>(a, n, T{0}, Add{}, threads);
}

template <class B, class T>
double mean(const T* a, std::size_t n, std::size_t threads = 0) {
    return n == 0 ? 0.0 : static_cast<double>(sum<B>(a, n, threads)) / static_cast<double>(n);
}


namespace detail {

inline constexpr std::size_t kMR = 4;    // строк C в микроядре
inline constexpr std::size_t kKC = 128;  // блок по K
inline constexpr std::size_t kNC = 256;  // блок по N

template <class B, class T, std::size_t MR>
inline void micro_kernel(const T* A, const T* Bm, T* C, std::size_t K, std::size_t N, std::size_t i,
                         std::size_t k0, std::size_t k1, std::size_t j0, std::size_t j1) {
    using R = typename B::template Reg<T>;
    constexpr std::size_t W = R::width;
    std::size_t j = j0;
    for (; j + W <= j1; j += W) {
        R c[MR];
        for (std::size_t r = 0; r < MR; ++r) c[r] = R::load(C + (i + r) * N + j);
        for (std::size_t k = k0; k < k1; ++k) {
            R b = R::load(Bm + k * N + j);
            for (std::size_t r = 0; r < MR; ++r) c[r] = c[r] + R::broadcast(A[(i + r) * K + k]) * b;
        }
        for (std::size_t r = 0; r < MR; ++r) c[r].store(C + (i + r) * N + j);
    }
    for (; j < j1; ++j) {
        for (std::size_t r = 0; r < MR; ++r) {
            T s = C[(i + r) * N + j];
            for (std::size_t k = k0; k < k1; ++k) s += A[(i + r) * K + k] * Bm[k * N + j];
            C[(i + r) * N + j] = s;
        }
    }
}

template <class B, class T>
void matmul_rows(const T* A, const T* Bm, T* C, std::size_t i0, std::size_t i1, std::size_t K,
                 std::size_t N) {
    std::fill(C + i0 * N, C + i1 * N, T{0});
    for (std::size_t k0 = 0; k0 < K; k0 += kKC) {
        const std::size_t k1 = std::min(K, k0 + kKC);
        for (std::size_t j0 = 0; j0 < N; j0 += kNC) {
            const std::size_t j1 = std::min(N, j0 + kNC);
            std::size_t i = i0;
            for (; i + kMR <= i1; i += kMR) micro_kernel<B, T, kMR>(A, Bm, C, K, N, i, k0, k1, j0, j1);
            for (; i < i1; ++i) micro_kernel<B, T, 1>(A, Bm, C, K, N, i, k0, k1, j0, j1);
        }
    }
}

}  // namespace detail

//  A_(M x K) * B_(K x N) = C_(M x N)
template <class B, class T>
void matmul(const T* A, const T* Bm, T* C, std::size_t M, std::size_t K, std::size_t N,
            std::size_t threads = 0) {
    constexpr std::size_t kWorkPerChunk = std::size_t{1} << 18;
    std::size_t grain = std::max<std::size_t>(detail::kMR, kWorkPerChunk / (K * N + 1));
    grain = (grain + detail::kMR - 1) / detail::kMR * detail::kMR;
    parallel_for(
        0, M, grain, threads,
        [&](std::size_t s, std::size_t e) { detail::matmul_rows<B>(A, Bm, C, s, e, K, N); }, detail::kMR);
}

}  // namespace hw
