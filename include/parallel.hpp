// Параллелизация на std::thread: постоянный пул потоков + parallel_for + parallel_reduce
// Исключения внутри задач не поддерживаются
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace hw {

class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    std::size_t max_threads() const { return workers_.size() + 1; }

    void run(std::size_t n_tasks, std::size_t threads, const std::function<void(std::size_t)>& fn) {
        threads = std::min({threads, max_threads(), n_tasks});
        if (threads <= 1) {
            for (std::size_t i = 0; i < n_tasks; ++i) fn(i);
            return;
        }
        std::lock_guard<std::mutex> region(run_mu_);
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_ = &fn;
            n_tasks_ = n_tasks;
            next_.store(0);
            helpers_ = threads - 1;
            pending_ = threads - 1;
            ++generation_;
        }
        cv_work_.notify_all();
        drain();
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&] { return pending_ == 0; });
        fn_ = nullptr;
    }

private:
    ThreadPool() {
        std::size_t n = std::thread::hardware_concurrency();
        if (const char* env = std::getenv("HW_THREADS")) {  // ручное задание числа потоков
            long v = std::strtol(env, nullptr, 10);
            if (v > 0) n = static_cast<std::size_t>(v);
        }
        if (n == 0) n = 1;
        for (std::size_t id = 0; id + 1 < n; ++id) workers_.emplace_back([this, id] { loop(id); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& t : workers_) t.join();
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void drain() {
        for (;;) {
            std::size_t i = next_.fetch_add(1);
            if (i >= n_tasks_) break;
            (*fn_)(i);
        }
    }

    void loop(std::size_t id) {
        std::size_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            if (id >= helpers_) continue;
            lk.unlock();
            drain();
            lk.lock();
            if (--pending_ == 0) cv_done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex run_mu_, mu_;
    std::condition_variable cv_work_, cv_done_;
    const std::function<void(std::size_t)>* fn_ = nullptr;
    std::size_t n_tasks_ = 0, helpers_ = 0, pending_ = 0, generation_ = 0;
    std::atomic<std::size_t> next_{0};
    bool stop_ = false;
};

// threads == 0 -> использовать все доступные потоки
inline std::size_t resolve_threads(std::size_t threads) {
    std::size_t m = ThreadPool::instance().max_threads();
    return threads == 0 ? m : std::min(threads, m);
}

namespace detail {
struct Split {
    std::size_t chunks, size;
};

inline Split split(std::size_t n, std::size_t grain, std::size_t threads, std::size_t align) {
    std::size_t chunks = std::min((n + grain - 1) / grain, threads * 4);
    std::size_t size = (n + chunks - 1) / chunks;
    size = (size + align - 1) / align * align;
    return {(n + size - 1) / size, size};
}
}  // namespace detail

// f(range_begin, range_end) вызывается для непересекающихся кусков [begin, end)
template <class F>
void parallel_for(std::size_t begin, std::size_t end, std::size_t grain, std::size_t threads, F&& f,
                  std::size_t align = 1) {
    if (end <= begin) return;
    const std::size_t n = end - begin;
    threads = resolve_threads(threads);
    if (threads == 1 || n <= grain) {
        f(begin, end);
        return;
    }
    auto s = detail::split(n, grain, threads, align);
    ThreadPool::instance().run(s.chunks, threads, [&](std::size_t c) {
        std::size_t b = begin + c * s.size;
        f(b, std::min(end, b + s.size));
    });
}

// range_fn(b, e) -> T считает частичный результат
// combine(T, T) -> T объединяет частичные результаты в порядке кусков
template <class T, class RangeFn, class Combine>
T parallel_reduce(std::size_t begin, std::size_t end, std::size_t grain, std::size_t threads, T identity,
                  RangeFn&& range_fn, Combine&& combine) {
    if (end <= begin) return identity;
    const std::size_t n = end - begin;
    threads = resolve_threads(threads);
    if (threads == 1 || n <= grain) return range_fn(begin, end);
    auto s = detail::split(n, grain, threads, 1);
    std::vector<T> partial(s.chunks, identity);
    ThreadPool::instance().run(s.chunks, threads, [&](std::size_t c) {
        std::size_t b = begin + c * s.size;
        partial[c] = range_fn(b, std::min(end, b + s.size));
    });
    T result = identity;
    for (const T& p : partial) result = combine(result, p);
    return result;
}

}  // namespace hw
