#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <memory>

#include "utils/base_config.h"
#include "utils/allocator.h"

namespace st {

namespace nn {
    class InitializerBase;
    class OptimizerBase;
}

class Storage {
public:
    explicit Storage(index_t size);
    Storage(const Storage& other, index_t offset);
    Storage(index_t size, data_t value);
    Storage(const data_t* data, index_t size);
    
    explicit Storage(const Storage& other) = default;
    explicit Storage(Storage&& other) = default;
    ~Storage() = default;
    Storage& operator=(const Storage& other) = delete;

    // inline function
    data_t operator[](index_t idx) const { return dptr_[idx]; }
    data_t& operator[](index_t idx) { return dptr_[idx]; }
    index_t offset(void) const { return dptr_ - bptr_->data_; }
    index_t version(void) const { return bptr_->version_; }
    void increment_version(void) const { ++bptr_->version_; }

    // friend function
    friend class nn::InitializerBase;
    friend class nn::OptimizerBase;
private:
    struct Vdata {
        index_t version_;
        data_t data_[1];
    };

    std::shared_ptr<Vdata> bptr_;  // base pointer
    data_t* dptr_;  // data pointer
};
}  // namespace st
#endif