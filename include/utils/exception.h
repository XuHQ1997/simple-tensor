#ifndef UTILS_EXCEPTION_H_
#define UTILS_EXCEPTION_H_

#include <cstdio>
#include <exception>
#include <algorithm>

namespace st {
namespace err {

struct Error: public std::exception {
    Error(const char* file, const char* func, unsigned int line);
    const char* what() const noexcept;

    static char msg_[300];
    const char* file_;
    const char* func_;
    const unsigned int line_;
};

}  // namespace err

#define ERROR_LOCATION __FILE__, __func__, __LINE__
#define THROW_ERROR(format, ...)	do {	\
    std::sprintf(err::Error::msg_, (format), ##__VA_ARGS__);    \
    throw err::Error(ERROR_LOCATION);	\
} while(0)

#ifndef CANCEL_CHECK
// base assert macro
#define CHECK_TRUE(expr, format, ...) \
    if(!(expr)) THROW_ERROR((format), ##__VA_ARGS__)

#define CHECK_NOT_NULL(ptr, format, ...) \
    if(nullptr == (ptr)) THROW_ERROR((format), ##__VA_ARGS__)

#define CHECK_EQUAL(x, y, format, ...) \
    if((x) != (y)) THROW_ERROR((format), ##__VA_ARGS__)

#define CHECK_IN_RANGE(x, lower, upper, format, ...) \
    if((x) < (lower) || (x) >= (upper)) THROW_ERROR((format), ##__VA_ARGS__)

#define CHECK_FLOAT_EQUAL(x, y, format, ...) \
    if(std::abs((x)-(y)) > 1e-4) THROW_ERROR((format), ##__VA_ARGS__) 

#define CHECK_INDEX_VALID(x, format, ...) \
    if((x) > INDEX_MAX) THROW_ERROR((format), ##__VA_ARGS__)

// assert macro only working for ExpImpl
#define CHECK_EXP_SAME_SHAPE(e1, e2) do {\
    CHECK_EQUAL((e1).ndim(), (e2).ndim(),  \
        "The dimension of operators, %dD and %dD , doesn't match.", \
        (e1).ndim(), (e2).ndim()); \
    for(index_t i = 0; i < (e1).ndim(); ++i) \
        CHECK_EQUAL((e1).size(i), (e2).size(i), \
            "The sizes on %d dimension, %d and %d, doesn't match.", \
            i, (e1).size(i), (e2).size(i)); \
} while(0)

#define CHECK_EXP_BROADCAST(e1, e2) do { \
    index_t min_dim = std::min((e1).ndim(), (e2).ndim()); \
    for(index_t i = 0; i < min_dim; ++i)  \
        CHECK_TRUE((e1).size(i) == (e2).size(i) || (e1).size(i) == 1 || (e2).size(i) == 1, \
            "The size on %d dimension, %d and %d, can't be broadcasted.", \
            i, (e1).size(i), (e2).size(i));  \
} while(0)

#else  // ifndef CANCEL_CHECK

#define CHECK_TRUE(expr, format, ...) {}
#define CHECK_NOT_NULL(ptr, format, ...) {}
#define CHECK_EQUAL(x, y, format, ...) {}
#define CHECK_IN_RANGE(x, lower, upper, format, ...) {}
#define CHECK_EXP_SAME_SHAPE(e1, e2) {}
#define CHECK_EXP_BROADCAST(e1, e2) {}

#endif

}  // namespace st
#endif
