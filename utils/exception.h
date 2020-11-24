#ifndef UTILS_EXCEPTION_H_
#define UTILS_EXCEPTION_H_

#include <cstdio>
#include <exception>

namespace st {
namespace err {

struct Error: public std::exception {
    Error(const char* file, const char* func, unsigned int line);
    const char* what() const noexcept;

    static char msg[300];
    const char* file;
    const char* func;
    const unsigned int line;
};

}  // namespace err

#define ERROR_LOCATION __FILE__, __func__, __LINE__
#define THROW_ERROR(format, ...)	do {	\
    std::sprintf(err::Error::msg, format, ##__VA_ARGS__);    \
    throw err::Error(ERROR_LOCATION);	\
} while(0)

// base assert macro
#define CHECK_NOT_NULL(ptr, format, ...) \
    if(nullptr == (ptr)) THROW_ERROR(format, ##__VA_ARGS__)

}  // namespace st
#endif
