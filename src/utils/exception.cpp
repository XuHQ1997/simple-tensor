#include <sstream>
#include <cstring>

#include "utils/exception.h"

namespace st {
namespace err {

char Error::msg_[300] = {0};

Error::Error(const char* file, const char* func, unsigned int line) 
        : file_(file), func_(func), line_(line) {};

const char* Error::what() const noexcept {
    std::stringstream s;
    s << std::endl;
    s << file_ << ":" << line_ << ": ";
    s << "In function " << func_ << "()." << std::endl;
    s << msg_;
    auto&& str = s.str();
    memcpy(msg_, str.c_str(), str.length());
    return msg_;
}

}  // namespace err
}  // namespace st