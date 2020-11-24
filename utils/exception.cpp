#include "exception.h"

namespace st {
namespace err {

char Error::msg[300] = {0};

Error::Error(const char* file, const char* func, unsigned int line) 
        : file(file), func(func), line(line) {};

const char* Error::what() const noexcept {
    return msg;
}

}  // namespace err
}  // namespace st