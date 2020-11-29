#include "utils/allocator.h"
#include "utils/exception.h"

namespace st {
inline unsigned int round_up(int x) { return x == 1 ? 1 : 1 << (64 - __builtin_clz(x-1)); }

Alloc& Alloc::self() {
    static Alloc alloc;
    return alloc;
}

void* Alloc::allocate(index_t size) {
    auto iter = self().cache_.find(size);
    void* res;
    if(iter != self().cache_.end()) {
        res = iter->second.release();
        self().cache_.erase(iter);
    } else {
        res = std::malloc(size);
        CHECK_NOT_NULL(res, "failed to allocate %d memory.", size);
    }
    return res;
}

void Alloc::deallocate(void* ptr, index_t size) {
    self().cache_.emplace(size, ptr);
}
} // namespace st