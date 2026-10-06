// Engram rows read with pread into caller memory, the GIL released for the whole call: first what the page cache holds
// without blocking, then the rest from the disk, each pass shared by a pool of threads when there is enough of it.

#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

using Reads = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

constexpr int64_t CACHED_PER_THREAD = 256;      // page-cache reads worth waking another thread for

// Bytes at the start of a read the page cache holds, taken without blocking (preadv2 RWF_NOWAIT).
static int64_t read_cached(int fd, int64_t offset, int64_t size, char* dst) {
    int64_t done = 0;
    while (done < size) {
        iovec io{dst + done, static_cast<size_t>(size - done)};
        const ssize_t got = preadv2(fd, &io, 1, offset + done, RWF_NOWAIT);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) break;                    // not cached, end of file or an error: read_one says which
        done += got;
    }
    return done;
}

// Size bytes of fd at offset into dst, short reads finished; "" or why it failed.
static std::string read_one(int fd, int64_t offset, int64_t size, char* dst) {
    int64_t done = 0;
    while (done < size) {
        const ssize_t got = pread(fd, dst + done, static_cast<size_t>(size - done), offset + done);
        if (got < 0 && errno == EINTR) continue;
        if (got < 0) return "pread of the Engram tables at byte " + std::to_string(offset + done) + ": " +
                            std::strerror(errno);
        if (got == 0) return "short read of the Engram tables at byte " + std::to_string(offset + done) +
                             ": the checkpoint changed";
        done += got;
    }
    return "";
}

// Threads that join a pass; the calling thread works too, so a pass never waits on a thread waking late.
class Pool {
 public:
    explicit Pool(int64_t threads) {
        TORCH_CHECK(threads >= 1, "an Engram read pool needs at least one thread");
        for (int64_t i = 1; i < threads; ++i) workers_.emplace_back([this] { serve(); });
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // Every (fd, offset, size, address) row of ``reads`` read before it returns -> "" or the first failure.
    std::string read(const Reads& reads) {
        TORCH_CHECK(reads.ndim() == 2 && reads.shape(1) == 4, "reads: int64 [n, 4] of (fd, offset, size, address)");
        const int64_t* rows = reads.data();
        const int64_t n = reads.shape(0);
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> one_call(call_);
        std::vector<int64_t> cached(n);
        pass(n, n / CACHED_PER_THREAD, [&](int64_t i) {
            const int64_t* r = rows + 4 * i;
            cached[i] = read_cached(static_cast<int>(r[0]), r[1], r[2], reinterpret_cast<char*>(r[3]));
            return std::string();
        });
        std::vector<int64_t> misses;
        for (int64_t i = 0; i < n; ++i) {
            if (cached[i] < rows[4 * i + 2]) misses.push_back(i);
        }
        const int64_t m = misses.size();
        return pass(m, m - 1, [&](int64_t k) {
            const int64_t* r = rows + 4 * misses[k];
            const int64_t done = cached[misses[k]];
            return read_one(static_cast<int>(r[0]), r[1] + done, r[2] - done, reinterpret_cast<char*>(r[3]) + done);
        });
    }

 private:
    // fn(0..n-1) on this thread and up to ``helpers`` woken ones -> "" or the first failure (which ends the pass).
    std::string pass(int64_t n, int64_t helpers, const std::function<std::string(int64_t)>& fn) {
        if (n <= 0) return "";
        const int64_t woken = std::clamp<int64_t>(helpers, 0, static_cast<int64_t>(workers_.size()));
        {
            std::lock_guard<std::mutex> lock(m_);
            fn_ = &fn;
            n_ = n;
            next_.store(0);
            error_.clear();
            want_ = woken;
        }
        for (int64_t i = 0; i < woken; ++i) wake_.notify_one();
        work();
        std::unique_lock<std::mutex> lock(m_);
        want_ = 0;                                  // threads not started yet stay out of this pass
        idle_.wait(lock, [this] { return active_ == 0; });
        return error_;
    }

    void serve() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(m_);
                wake_.wait(lock, [this] { return stop_ || want_ > 0; });
                if (stop_) return;
                --want_;
                ++active_;
            }
            work();
            std::lock_guard<std::mutex> lock(m_);
            if (--active_ == 0) idle_.notify_all();
        }
    }

    void work() {
        for (int64_t i = next_.fetch_add(1); i < n_; i = next_.fetch_add(1)) {
            std::string err = (*fn_)(i);
            if (!err.empty()) {
                std::lock_guard<std::mutex> lock(m_);
                if (error_.empty()) error_ = std::move(err);
                next_.store(n_);                    // hand out no more
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex call_, m_;
    std::condition_variable wake_, idle_;
    const std::function<std::string(int64_t)>* fn_ = nullptr;
    int64_t n_ = 0;
    std::atomic<int64_t> next_{0};
    int64_t want_ = 0, active_ = 0;
    bool stop_ = false;
    std::string error_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Pool>(m, "Pool")
        .def(py::init<int64_t>(), py::arg("threads"))
        .def("read", &Pool::read, py::arg("reads"));
}
