#include "utils/allocator.h"
#include "utils/exception.h"

#include <iostream>

namespace st {
inline unsigned int round_up(int x) { return x == 1 ? 1 : 1 << (64 - __builtin_clz(x-1)); }

index_t Alloc::allocate_memory_size;
index_t Alloc::deallocate_memory_size;

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
    allocate_memory_size += size;
    return res;
}

void Alloc::deallocate(void* ptr, index_t size) {
    deallocate_memory_size += size;
    self().cache_.emplace(size, ptr);
}

bool Alloc::all_clear() {
    return allocate_memory_size == deallocate_memory_size;
}

} // namespace st