#pragma once

#include "ggml.h"

#include <cstdarg>

#ifndef __GNUC__
#    define LOG_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__)
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#ifndef LOG_VERBOSITY
#define LOG_VERBOSITY 10
#endif

struct gpt_log;

struct gpt_log * gpt_log_init();
struct gpt_log * gpt_log_main();
void             gpt_log_pause (struct gpt_log * log);
void             gpt_log_resume(struct gpt_log * log);
void             gpt_log_free  (struct gpt_log * log);

LOG_ATTRIBUTE_FORMAT(4, 5)
void gpt_log_add(struct gpt_log * log, enum ggml_log_level level, int verbosity, const char * fmt, ...);

void gpt_log_set_file      (struct gpt_log * log, const char * file);
void gpt_log_set_timestamps(struct gpt_log * log, bool timestamps);

#define LOG_TMPL(level, verbosity, ...) \
    do { \
        if ((verbosity) <= LOG_VERBOSITY) { \
            gpt_log_add(gpt_log_main(), (level), (verbosity), __VA_ARGS__); \
        } \
    } while (0)

#define LOG(...)             LOG_TMPL(GGML_LOG_LEVEL_NONE, 0,         __VA_ARGS__)
#define LOGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_NONE, verbosity, __VA_ARGS__)

#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  0, __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  0, __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, 0, __VA_ARGS__)
#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, 0, __VA_ARGS__)

#define LOG_INFV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  verbosity, __VA_ARGS__)
#define LOG_WRNV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  verbosity, __VA_ARGS__)
#define LOG_ERRV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, verbosity, __VA_ARGS__)
#define LOG_DBGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, verbosity, __VA_ARGS__)

#define LOG_TOKENS_TOSTR_PRETTY(...) std::string("dummy")
