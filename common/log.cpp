#include "log.h"

#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

int gpt_log_verbosity_env = getenv("LLAMA_LOG") ? atoi(getenv("LLAMA_LOG")) : LOG_DEFAULT_LLAMA;

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

    int64_t timestamp;

    std::vector<char> msg;

    // signals the worker thread to stop
    bool is_end;

    void print(FILE * file = nullptr) const {
        FILE * fcur = file;
        if (!fcur) {
            // stderr displays DBG messages only when the verbosity is high
            // these messages can still be logged to a file
            if (level == GGML_LOG_LEVEL_DEBUG && gpt_log_verbosity_env < LOG_DEFAULT_DEBUG) {
                return;
            }

            fcur = stdout;

            if (level != GGML_LOG_LEVEL_NONE) {
                fcur = stderr;
            }
        }

        if (level != GGML_LOG_LEVEL_NONE) {
            if (timestamp) {
                // [M.s.ms.us]
                fprintf(fcur, "" LOG_COL_BLUE "%05d.%02d.%03d.%03d" LOG_COL_DEFAULT " ",
                        (int) (timestamp / 1000000 / 60),
                        (int) (timestamp / 1000000 % 60),
                        (int) (timestamp / 1000 % 1000),
                        (int) (timestamp % 1000));
            }

            switch (level) {
                case GGML_LOG_LEVEL_INFO:  fprintf(fcur, LOG_COL_GREEN   "I " LOG_COL_DEFAULT); break;
                case GGML_LOG_LEVEL_WARN:  fprintf(fcur, LOG_COL_MAGENTA "W "                ); break;
                case GGML_LOG_LEVEL_ERROR: fprintf(fcur, LOG_COL_RED     "E "                ); break;
                case GGML_LOG_LEVEL_DEBUG: fprintf(fcur, LOG_COL_YELLOW  "D "                ); break;
                default:
                    break;
            }
        }

        fprintf(fcur, "%s", msg.data());

        if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_DEBUG) {
            fprintf(fcur, "%s", LOG_COL_DEFAULT);
        }

        fflush(fcur);
    }
};

struct gpt_log {
    gpt_log(size_t capacity) {
        file = nullptr;
        timestamps = true;
        running = false;
        t_start = t_us();
        entries.resize(capacity);
        for (auto & entry : entries) {
            entry.msg.resize(256);
        }
        head = 0;
        tail = 0;

        resume();
    }

    ~gpt_log() {
        pause();
        if (file) {
            fclose(file);
        }
    }

private:
    std::mutex mtx;
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

    // worker thread copies into this
    gpt_log_entry cur;

public:
    void add(enum ggml_log_level level, const char * fmt, va_list args) {
        std::lock_guard<std::mutex> lock(mtx);

        if (!running) {
            return;
        }

        auto & entry = entries[tail];

        {
            // cannot use args twice, so make a copy in case we need to expand the buffer
            va_list args_copy;
            va_copy(args_copy, args);

#if 1
            const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args);
            if (n >= entry.msg.size()) {
                entry.msg.resize(n + 1);
                vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args_copy);
            }
#else
            // hack for bolding arguments

            std::stringstream ss;
            for (int i = 0; fmt[i] != 0; i++) {
                if (fmt[i] == '%') {
                    ss << LOG_COL_BOLD;
                    while (fmt[i] != ' ' && fmt[i] != ')' && fmt[i] != ']' && fmt[i] != 0) ss << fmt[i++];
                    ss << LOG_COL_DEFAULT;
                    if (fmt[i] == 0) break;
                }
                ss << fmt[i];
            }
            const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), ss.str().c_str(), args);
            if (n >= entry.msg.size()) {
                entry.msg.resize(n + 1);
                vsnprintf(entry.msg.data(), entry.msg.size(), ss.str().c_str(), args_copy);
            }
#endif
        }

        entry.level = level;
        entry.timestamp = 0;
        if (timestamps) {
            entry.timestamp = t_us() - t_start;
        }
        entry.is_end = false;

        tail = (tail + 1) % entries.size();
        if (tail == head) {
            // expand the buffer
            std::vector<gpt_log_entry> new_entries(2*entries.size());

            size_t new_tail = 0;

            do {
                new_entries[new_tail] = std::move(entries[head]);

                head     = (head     + 1) % entries.size();
                new_tail = (new_tail + 1);
            } while (head != tail);

            head = 0;
            tail = new_tail;

            for (size_t i = tail; i < new_entries.size(); i++) {
                new_entries[i].msg.resize(256);
            }

            entries = std::move(new_entries);
        }

        cv.notify_one();
    }

    void resume() {
        std::lock_guard<std::mutex> lock(mtx);

        if (running) {
            return;
        }

        running = true;

        thrd = std::thread([this]() {
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [this]() { return head != tail; });

                    cur = entries[head];

                    head = (head + 1) % entries.size();
                }

                if (cur.is_end) {
                    break;
                }

                cur.print(); // stdout and stderr

                if (file) {
                    cur.print(file);
                }
            }
        });
    }

    void pause() {
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (!running) {
                return;
            }

            running = false;

            auto & entry = entries[tail];

            entry.is_end = true;

            tail = (tail + 1) % entries.size();

            cv.notify_one();
        }

        thrd.join();
    }

    void set_file(const char * path) {
        pause();

        if (file) {
            fclose(file);
        }

        if (path) {
            file = fopen(path, "w");
        } else {
            file = nullptr;
        }

        resume();
    }

    void set_timestamps(bool timestamps) {
        std::lock_guard<std::mutex> lock(mtx);

        this->timestamps = timestamps;
    }
};

struct gpt_log * gpt_log_init() {
    return new gpt_log{256};
}

struct gpt_log * gpt_log_main() {
    static struct gpt_log log{256};

    return &log;
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

void gpt_log_add(struct gpt_log * log, enum ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log->add(level, fmt, args);
    va_end(args);
}

void gpt_log_set_file(struct gpt_log * log, const char * file) {
    log->set_file(file);
}

void gpt_log_set_timestamps(struct gpt_log * log, bool timestamps) {
    log->set_timestamps(timestamps);
}
