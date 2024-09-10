#include "log.h"

#include <thread>
#include <mutex>
#include <cstdio>
#include <condition_variable>

#define LOG_MAX_MESSAGE_SIZE (256)

#define LOG_COLORS // TMP

#ifdef LOG_COLORS
#define LOG_COL_DEFAULT "\033[0m"
#define LOG_COL_BOLD    "\033[1m"
#define LOG_COL_RED     "\033[31m"
#define LOG_COL_GREEN   "\033[32m"
#define LOG_COL_YELLOW  "\033[33m"
#define LOG_COL_BLUE    "\033[34m"
#define LOG_COL_MAGENTA "\033[35m"
#define LOG_COL_CYAN    "\033[36m"
#define LOG_COL_WHITE   "\033[37m"
#else
#define LOG_COL_DEFAULT ""
#define LOG_COL_BOLD    ""
#define LOG_COL_RED     ""
#define LOG_COL_GREEN   ""
#define LOG_COL_YELLOW  ""
#define LOG_COL_BLUE    ""
#define LOG_COL_MAGENTA ""
#define LOG_COL_CYAN    ""
#define LOG_COL_WHITE   ""
#endif

static int64_t t_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

struct gpt_log_entry {
    enum ggml_log_level level;

    int verbosity;
    int64_t timestamp;

    // static-sized message
    char msg_stck[LOG_MAX_MESSAGE_SIZE];

    // if it doesn't fit in the stack, it goes here
    std::vector<char> msg_heap;

    // signals the worker thread to stop
    bool is_end;

    void print(FILE * file) {
        if (level != GGML_LOG_LEVEL_NONE) {
            if (timestamp) {
                // [M.s.ms.us]
                fprintf(file, "[%04d.%02d.%03d.%03d] ",
                        (int) (timestamp / 1000000 / 60),
                        (int) (timestamp / 1000000 % 60),
                        (int) (timestamp / 1000 % 1000),
                        (int) (timestamp % 1000));
            }

            switch (level) {
                case GGML_LOG_LEVEL_INFO:
                    fprintf(file, LOG_COL_GREEN "INF " LOG_COL_DEFAULT);
                    break;
                case GGML_LOG_LEVEL_WARN:
                    fprintf(file, LOG_COL_MAGENTA "WRN " LOG_COL_DEFAULT);
                    break;
                case GGML_LOG_LEVEL_ERROR:
                    fprintf(file, LOG_COL_RED "ERR " LOG_COL_DEFAULT);
                    break;
                case GGML_LOG_LEVEL_DEBUG:
                    fprintf(file, LOG_COL_YELLOW "DBG " LOG_COL_DEFAULT);
                    break;
                default:
                    break;
            }
        }

        if (msg_heap.empty()) {
            fprintf(file, "%s", msg_stck);
        } else {
            fprintf(file, "%s", msg_heap.data());
        }

        fflush(file);
    }
};

struct gpt_log {
    gpt_log(size_t capacity) {
        file = nullptr;
        timestamps = true;
        running = false;
        t_start = t_us();
        entries.resize(capacity);
        head = 0;
        tail = 0;
        buffer.resize(1024);

        resume();
    }

    ~gpt_log() {
        pause();
        if (file) {
            fclose(file);
        }
    }

private:
    std::mutex mtx_inp;
    std::mutex mtx_wrk;

    std::thread thrd;
    std::condition_variable cv;

    FILE * file;

    bool timestamps;
    bool running;

    int64_t t_start;

    // ring buffer of entries
    std::vector<gpt_log_entry> entries;
    size_t head;
    size_t tail;

    // print the message here before pushing
    std::vector<char> buffer;

public:
    void add(enum ggml_log_level level, int verbosity, const char * fmt, va_list args) {
        std::unique_lock<std::mutex> lock_inp(mtx_inp);

        if (!running) {
            return;
        }

        const size_t n = vsnprintf(buffer.data(), buffer.size(), fmt, args);
        if (n >= buffer.size()) {
            buffer.resize(n + 1);
            vsnprintf(buffer.data(), buffer.size(), fmt, args);
        }

        std::lock_guard<std::mutex> lock_wrk(mtx_wrk);

        auto & entry = entries[tail];

        if (n < LOG_MAX_MESSAGE_SIZE) {
            memcpy(entry.msg_stck, buffer.data(), n);
            entry.msg_stck[n] = '\0';
            entry.msg_heap.clear();
        } else {
            entry.msg_heap.resize(n + 1);
            memcpy(entry.msg_heap.data(), buffer.data(), n);
            entry.msg_heap[n] = '\0';
        }

        lock_inp.unlock();

        entry.level = level;
        entry.verbosity = verbosity;
        entry.timestamp = 0;
        if (timestamps) {
            entry.timestamp = t_us() - t_start;
        }
        entry.is_end = false;

        tail = (tail + 1) % entries.size();
        if (tail == head) {
            // expand the buffer
            size_t new_size = entries.size() * 2;
            std::vector<gpt_log_entry> new_entries(new_size);

            size_t new_head = 0;
            size_t new_tail = 0;

            while (head != tail) {
                new_entries[new_tail] = entries[head];

                head = (head + 1) % entries.size();
                new_tail = (new_tail + 1) % new_size;
            }

            head = new_head;
            tail = new_tail;

            entries = std::move(new_entries);
        }

        cv.notify_one();
    }

    void resume() {
        std::lock_guard<std::mutex> lock_inp(mtx_inp);

        if (running) {
            return;
        }

        running = true;

        thrd = std::thread([this]() {
            while (true) {
                std::unique_lock<std::mutex> lock_wrk(mtx_wrk);
                cv.wait(lock_wrk, [this]() { return head != tail; });

                auto & entry = entries[head];

                if (entry.is_end) {
                    break;
                }

                entry.print(stdout);

                if (file) {
                    entry.print(file);
                }

                head = (head + 1) % entries.size();
            }
        });
    }

    void pause() {
        std::lock_guard<std::mutex> lock_inp(mtx_inp);

        if (!running) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock_wrk(mtx_wrk);

            auto & entry = entries[tail];

            entry.is_end = true;

            tail = (tail + 1) % entries.size();

            cv.notify_one();
        }

        thrd.join();

        running = false;
    }

    void set_file(const char * path) {
        std::lock_guard<std::mutex> lock_inp(mtx_inp);
        std::lock_guard<std::mutex> lock_wrk(mtx_wrk);

        if (file) {
            fclose(file);
        }

        if (path) {
            file = fopen(path, "w");
        } else {
            file = nullptr;
        }
    }

    void set_timestamps(bool timestamps) {
        std::lock_guard<std::mutex> lock_inp(mtx_inp);
        std::lock_guard<std::mutex> lock_wrk(mtx_wrk);

        this->timestamps = timestamps;
    }
};

struct gpt_log * gpt_log_init() {
    return new gpt_log{1024};
}

struct gpt_log * gpt_log_main() {
    static struct gpt_log * log = gpt_log_init();
    return log;
}

void gpt_log_pause(struct gpt_log * log) {
    log->pause();
}

void gpt_log_resume(struct gpt_log * log) {
    log->resume();
}

void gpt_log_free(struct gpt_log * log) {
    delete log;
}

void gpt_log_add(struct gpt_log * log, enum ggml_log_level level, int verbosity, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log->add(level, verbosity, fmt, args);
    va_end(args);
}

void gpt_log_set_file(struct gpt_log * log, const char * file) {
    log->set_file(file);
}

void gpt_log_set_timestamps(struct gpt_log * log, bool timestamps) {
    log->set_timestamps(timestamps);
}
