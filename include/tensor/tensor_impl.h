#ifndef TENSOR_TENSOR_IMPL_H
#define TENSOR_TENSOR_IMPL_H

#include "tensor/storage.h"
#include "tensor/shape.h"

namespace st {

class Tensor {
public:
    // constructor
    Tensor(const Storage& storage, const Shape& shape, const IndexArray& stride,
           bool requires_grad=false);
    Tensor(const Storage& storage, const Shape& shape, 
           bool requires_grad=false);
    Tensor(const data_t* data, const Shape& shape, 
           bool requires_grad=false);
    explicit Tensor(const Shape& shape, 
                    bool requires_grad=false);
    Tensor(Storage&& storage, Shape&& shape, IndexArray&& stride, 
           bool requires_grad=false);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;

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

    Tensor slice(index_t idx, index_t dim=0) const;
    Tensor slice(index_t start_idx, index_t end_idx, index_t dim) const;
    Tensor transpose(index_t dim1, index_t dim2) const;
    Tensor view(const Shape& shape) const;
    Tensor squeeze(void) const;
    Tensor unsqueeze(index_t dim) const;

    // friend function
    friend std::ostream& operator<<(std::ostream& out, const Tensor& t);
private:
    Storage storage_;
    Shape shape_;
    IndexArray stride_;

    bool requires_grad_;
};

}  // namespace st

#endif