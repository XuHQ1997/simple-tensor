#ifndef TENSOR_TENSOR_IMPL_H
#define TENSOR_TENSOR_IMPL_H

#include "exp/exp_impl.h"
#include "tensor/storage.h"
#include "tensor/shape.h"
#include "utils/exception.h"


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
    TensorImpl& operator=(const TensorImpl& other);

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

    // member function for expression template
    data_t eval(IndexArray& inds) const;
    data_t eval(index_t idx) const;
    template<typename ImplType> TensorImpl& operator=(const ImplType& exp_impl);
    template<typename ImplType> TensorImpl& operator+=(const ImplType& exp_impl);

    // friend function
    friend std::ostream& operator<<(std::ostream& out, const TensorImpl& t);
private:
    template<typename ImplType> 
    TensorImpl& __assign(const ImplType& exp_impl);
    template<typename ImplType> 
    TensorImpl& __inplacement_add(const ImplType& exp_impl);

    Storage storage_;
    Shape shape_;
    IndexArray stride_;

    bool requires_grad_;
};


// member template function definition
template<typename ImplType> 
TensorImpl& TensorImpl::operator=(const ImplType& exp_impl) {
    CHECK_TRUE(is_contiguous(), "operator= is only supported for contiguous Tensor.");
    CHECK_EXP_BROADCAST(*this, exp_impl);
    return __assign(exp_impl);
}

template<typename ImplType>
TensorImpl& TensorImpl::operator+=(const ImplType& exp_impl) {
    CHECK_TRUE(is_contiguous(), "operator+= is only supported for contiguous Tensor.");
    CHECK_EXP_BROADCAST(*this, exp_impl);
    return __inplacement_add(exp_impl);
}

template<typename ImplType>
TensorImpl& TensorImpl::__assign(const ImplType& exp_impl) {
    IndexArray inds(ndim());
    for(index_t i = 0; i < shape_.dsize(); ++i) {
        for(index_t ii = i, j = 0; j < ndim(); ++j) {
            if(stride_[j] != 0) {
                inds[j] = ii / stride_[j];
                ii %= stride_[j];
            }
        }
        storage_[i] = exp_impl.eval(inds);
    }
    return *this;
}

template<typename ImplType>
TensorImpl& TensorImpl::__inplacement_add(const ImplType& exp_impl) {
    IndexArray inds(ndim());
    for(index_t i = 0; i < shape_.dsize(); ++i) {
        for(index_t ii = i, j = 0; j < ndim(); ++j) {
            if(stride_[j] != 0) {
                inds[j] = ii / stride_[j];
                ii %= stride_[j];
            }
        }
        storage_[i] += exp_impl.eval(inds);
    }
    return *this;
}

}  // namespace st

#endif