#include "utils/exception.h"

namespace st {
namespace err {

char Error::msg_[300] = {0};

Error::Error(const char* file, const char* func, unsigned int line) 
        : file_(file), func_(func), line_(line) {};

const char* Error::what() const noexcept {
    return msg_;
}

}  // namespace err
}  // namespace st