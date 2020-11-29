#include <cstring>

#include "tensor/storage.h"

namespace st {

Storage::Storage(index_t size)
        : bptr_(Alloc::shared_allocate<Vdata>(size * sizeof(data_t) + sizeof(index_t))),
          dptr_(bptr_->data_) {
    bptr_->version_ = 0;
}

Storage::Storage(const Storage& other, index_t offset)
        : bptr_(other.bptr_),
          dptr_(other.dptr_ + offset) {}

Storage::Storage(index_t size, data_t value)
        : Storage(size) {
    std::memset(dptr_, value, size * sizeof(data_t));
}

Storage::Storage(const data_t* data, index_t size)
        : Storage(size) {
    std::memcpy(dptr_, data, size * sizeof(data_t));
}

}  // namespace st