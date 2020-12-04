#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <memory>

#include "exp/exp.h"
#include "exp/exp_impl.h"
#include "tensor/tensor_impl.h"
#include "exp/operator/function.h"

namespace st {

template class Exp<TensorImpl>;
template class ExpImplPtr<TensorImpl>;


// Shell of TensorImpl. See exp/exp.h for more information.
class Tensor : public Exp<TensorImpl> {
public:
    Tensor(const Storage& storage, const Shape& shape, const IndexArray& stride,
           bool requires_grad=false);
    Tensor(const Storage& storage, const Shape& shape, 
           bool requires_grad=false);
    Tensor(const data_t* data, const Shape& shape, 
           bool requires_grad=false);
    explicit Tensor(const Shape& shape, 
                    bool requires_grad=false);
    Tensor(TensorImpl&& impl);
    Tensor(Alloc::NontrivialUniquePtr<TensorImpl>&& ptr);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;

    index_t ndim(void) const;
    index_t size(index_t idx) const;
    const Shape& size(void) const;
    index_t offset(void) const;
    const IndexArray& stride(void) const;
    index_t version(void) const;
    
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

    template<typename ImplType> Tensor& operator=(const Exp<ImplType>& exp);
    template<typename ImplType> Tensor& operator+=(const Exp<ImplType>& exp);

    // friend function
    friend std::ostream& operator<<(std::ostream& out, const Tensor& t);
};

template<typename ImplType> Tensor& Tensor::operator=(const Exp<ImplType>& exp) {
    impl_ptr_->operator=(*exp.impl_ptr());
    return *this;
}

template<typename ImplType> Tensor& Tensor::operator+=(const Exp<ImplType>& exp) {
    impl_ptr_->operator+=(*exp.impl_ptr());
    return *this;
}

}  // namespace st
#endif
