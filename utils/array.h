#ifndef UTILS_ARRAY_H
#define UTILS_ARRAY_H

#include <initializer_list>

#include "utils/base_config.h"
#include "utils/allocator.h"

namespace st {

template<typename Dtype>
class DynamicArray {
public:
    DynamicArray(index_t size_) : size(size_), dptr(Alloc::allocate(size)) 
        {};
    DynamicArray(std::initializer_list<Dtype> data) : DynamicArray(data.size()) {
        for(int i = 0; i < data.size(); ++i)
            dptr[i] = data[i];
    }
    DynamicArray(const DynamicArray<Dtype>& other) : DynamicArray(other.size()) {
        memcpy(dptr, other.dptr, size);
    }
    ~DynamicArray() { Alloc::deallocate(dptr, size); }

    Dtype operator[](index_t idx) { return dptr[idx]; }
    Dtype operator[](index_t idx) const { return dptr[idx]; }
    index_t size() const { return size; }
private:
    index_t size;
    Dtype* dptr;
};


} // namespace st

#endif