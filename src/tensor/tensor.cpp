#include "tensor/tensor.h"

namespace st {

Tensor::Tensor(const Storage& storage, const Shape& shape, const IndexArray& stride,
               bool requires_grad)
        : Exp<TensorImpl>(
            Alloc::unique_construct<TensorImpl>(storage, shape, stride, requires_grad))
    {}

Tensor::Tensor(const Storage& storage, const Shape& shape, 
               bool requires_grad)
        : Exp<TensorImpl>(
            Alloc::unique_construct<TensorImpl>(storage, shape, requires_grad))
    {}

Tensor::Tensor(const data_t* data, const Shape& shape, 
               bool requires_grad)
        : Exp<TensorImpl>(
            Alloc::unique_construct<TensorImpl>(data, shape, requires_grad))
    {}

Tensor::Tensor(const Shape& shape, bool requires_grad)
        : Exp<TensorImpl>(
            Alloc::unique_construct<TensorImpl>(shape, requires_grad))
    {}

Tensor::Tensor(TensorImpl&& impl)
        : Exp<TensorImpl>(
            Alloc::unique_construct<TensorImpl>(std::move(impl)))
    {}

Tensor::Tensor(Alloc::NontrivialUniquePtr<TensorImpl>&& ptr) 
        : Exp<TensorImpl>(std::move(ptr))
    {}

index_t Tensor::ndim(void) const { return impl_ptr_->ndim(); }
index_t Tensor::size(index_t idx) const { return impl_ptr_->size(idx); }
const Shape& Tensor::size(void) const { return impl_ptr_->size(); }
index_t Tensor::offset(void) const { return impl_ptr_->offset(); }
const IndexArray& Tensor::stride(void) const { return impl_ptr_->stride(); }
index_t Tensor::version(void) const { return impl_ptr_->version(); }

bool Tensor::is_contiguous(void) const { return impl_ptr_->is_contiguous(); }

data_t& Tensor::operator[](std::initializer_list<index_t> ids) {
    return impl_ptr_->operator[](ids);
};
data_t Tensor::operator[](std::initializer_list<index_t> ids) const {
    return impl_ptr_->operator[](ids);
}
data_t Tensor::item(void) const { return impl_ptr_->item(); }

Tensor Tensor::slice(index_t idx, index_t dim) const { 
    return Tensor(impl_ptr_->slice(idx, dim)); 
}
Tensor Tensor::slice(index_t start_idx, index_t end_idx, index_t dim) const {
    return Tensor(impl_ptr_->slice(start_idx, end_idx, dim));
}
Tensor Tensor::transpose(index_t dim1, index_t dim2) const {
    return Tensor(impl_ptr_->transpose(dim1, dim2));
}
Tensor Tensor::view(const Shape& shape) const {
    return Tensor(impl_ptr_->view(shape));
}
Tensor Tensor::squeeze(void) const {
    return Tensor(impl_ptr_->squeeze());
}
Tensor Tensor::unsqueeze(index_t dim) const {
    return Tensor(impl_ptr_->unsqueeze(dim));
}

// friend function
std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    return out << *(t.impl_ptr_);
}

}  // namespace st
