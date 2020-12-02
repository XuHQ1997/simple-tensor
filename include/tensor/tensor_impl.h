#ifndef TENSOR_TENSOR_IMPL_H
#define TENSOR_TENSOR_IMPL_H

#include "exp/exp.h"
#include "tensor/storage.h"
#include "tensor/shape.h"

namespace st {

class TensorImpl : public ExpImpl<TensorImpl> {
public:
    // constructor
    TensorImpl(const Storage& storage, const Shape& shape, const IndexArray& stride,
           bool requires_grad=false);
    TensorImpl(const Storage& storage, const Shape& shape, 
           bool requires_grad=false);
    TensorImpl(const data_t* data, const Shape& shape, 
           bool requires_grad=false);
    explicit TensorImpl(const Shape& shape, 
                    bool requires_grad=false);
    TensorImpl(Storage&& storage, Shape&& shape, IndexArray&& stride, 
           bool requires_grad=false);
    TensorImpl(const TensorImpl& other) = default;
    TensorImpl(TensorImpl&& other) = default;

    // inline function
    index_t ndim(void) const { return shape_.ndim(); }
    index_t size(index_t idx) const { return shape_[idx]; }
    const Shape& size(void) const { return shape_; }
    index_t offset(void) const { return storage_.offset(); }
    const IndexArray& stride(void) const { return stride_; }
    index_t version(void) const { return storage_.version(); }
    
    // other method
    bool is_contiguous(void) const;
    
    data_t& operator[](std::initializer_list<index_t> ids);
    data_t operator[](std::initializer_list<index_t> ids) const;
    data_t item(void) const;

    Alloc::NontrivialUniquePtr<TensorImpl> slice(index_t idx, index_t dim=0) const;
    Alloc::NontrivialUniquePtr<TensorImpl> slice(index_t start_idx, 
                                                 index_t end_idx, index_t dim) const;
    Alloc::NontrivialUniquePtr<TensorImpl> transpose(index_t dim1, index_t dim2) const;
    Alloc::NontrivialUniquePtr<TensorImpl> view(const Shape& shape) const;
    Alloc::NontrivialUniquePtr<TensorImpl> squeeze(void) const;
    Alloc::NontrivialUniquePtr<TensorImpl> unsqueeze(index_t dim) const;

    // friend function
    friend std::ostream& operator<<(std::ostream& out, const TensorImpl& t);
private:
    Storage storage_;
    Shape shape_;
    IndexArray stride_;

    bool requires_grad_;
};

}  // namespace st

#endif