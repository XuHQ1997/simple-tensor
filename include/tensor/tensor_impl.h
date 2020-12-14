#ifndef TENSOR_TENSOR_IMPL_H
#define TENSOR_TENSOR_IMPL_H

#include <initializer_list>
#include <utility>

#include "exp/exp_impl.h"
#include "tensor/storage.h"
#include "tensor/shape.h"
#include "utils/exception.h"


namespace st {

// foward declaration
struct AutoGradMeta;

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
    
    TensorImpl(const TensorImpl& other) = delete;
    TensorImpl(TensorImpl&& other) = default;
    TensorImpl& operator=(const TensorImpl& other);

    // inline function
    index_t ndim(void) const { return shape_.ndim(); }
    index_t size(index_t idx) const { return shape_[idx]; }
    const Shape& size(void) const { return shape_; }
    index_t offset(void) const { return storage_.offset(); }
    const IndexArray& stride(void) const { return stride_; }
    index_t version(void) const { return storage_.version(); }
    bool requires_grad(void) const { return requires_grad_; }

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
    Alloc::NontrivialUniquePtr<TensorImpl> 
    permute(std::initializer_list<index_t> dims) const;

    // member function for expression template
    data_t eval(IndexArray& inds) const;
    data_t eval(index_t idx) const;
    template<typename ImplType> TensorImpl& operator=(const ImplType& exp_impl);
    template<typename ImplType> TensorImpl& operator+=(const ImplType& exp_impl);

    // friend function
    friend std::ostream& operator<<(std::ostream& out, const TensorImpl& t);
    friend ExpImplPtr<TensorImpl>;
private:
    template<typename ImplType> 
    TensorImpl& __assign(const ImplType& exp_impl);
    template<typename ImplType> 
    TensorImpl& __inplacement_add(const ImplType& exp_impl);
    template<typename ImplType> 
    TensorImpl& __assign_uncontiguous(const ImplType& exp_impl);
    template<typename ImplType> 
    TensorImpl& __inplacement_add_uncontiguous(const ImplType& exp_impl);
    
    template<typename ImplType> void backward(const ImplType& grad);
    void backward(void);

    Storage storage_;
    Shape shape_;
    IndexArray stride_;

    bool requires_grad_;
    Alloc::NontrivialUniquePtr<AutoGradMeta> gradmeta_ptr_;
};

// Template specialization for ExpImplPtr
template<> 
class ExpImplPtr<TensorImpl> {
public:
    ExpImplPtr(Alloc::NontrivialUniquePtr<TensorImpl>&& ptr, bool with_grad)
            : ptr_(ptr.release()),
              with_grad_(with_grad && static_cast<TensorImpl*>(ptr_)->requires_grad()),
              version_(static_cast<TensorImpl*>(ptr_)->version()) {
        increment_counters();
    }
    ExpImplPtr(const TensorImpl& impl, bool with_grad)
            : ptr_(const_cast<TensorImpl*>(&impl)),
              with_grad_(with_grad && static_cast<TensorImpl*>(ptr_)->requires_grad()),
              version_(static_cast<TensorImpl*>(ptr_)->version()) {
        increment_counters();
    }
    ExpImplPtr(const ExpImplPtr& other, bool with_grad)
            : ptr_(other.ptr_),
              with_grad_(with_grad && static_cast<TensorImpl*>(ptr_)->requires_grad()),
              version_(static_cast<TensorImpl*>(ptr_)->version()) { 
        increment_counters(); 
    }
    ~ExpImplPtr() { decrease_refcount(); }

    TensorImpl* operator->(void) const { return static_cast<TensorImpl*>(ptr_); }
    const TensorImpl& operator*(void) const { return *static_cast<TensorImpl*>(ptr_); }
    explicit operator bool() const { return ptr_ != nullptr; }

    template<typename GradImplType>
    void invoke_backward(const GradImplType& grad) {
        TensorImpl* ptr = static_cast<TensorImpl*>(ptr_);
        if(with_grad_) {
            CHECK_EQUAL(version_, ptr->version(),
                "Leaf variable has been moved into the graph interior");
            -- ptr->gradcount_;
            ptr->backward(grad);
        }
    }

    void invoke_backward(void) {
        TensorImpl* ptr = static_cast<TensorImpl*>(ptr_);
        if(with_grad_) {
            CHECK_EQUAL(version_, ptr->version(),
                "Leaf variable has been moved into the graph interior");
            -- ptr->gradcount_;
            ptr->backward();
        }
    }
private:
    void increment_counters() { 
        ++ ptr_->refcount_; 
        if(with_grad_)
            ++ ptr_->gradcount_;
    }

    void decrease_refcount() {
        -- ptr_->refcount_;
        if(ptr_->refcount_ == 0)
            delete_handler(static_cast<void*>(ptr_));
    }


    ExpImpl<TensorImpl>* ptr_;
    bool with_grad_;
    index_t version_;
    Alloc::nontrivial_delete_handler<TensorImpl> delete_handler;
};
}  // namespace st


#include "tensor/grad_meta.h"

namespace st {

// member template function definition
template<typename ImplType> 
TensorImpl& TensorImpl::operator=(const ImplType& exp_impl) {
    CHECK_EXP_BROADCAST(*this, exp_impl);
    storage_.increment_version();
    if(is_contiguous())
        return __assign(exp_impl);
    else    
        return __assign_uncontiguous(exp_impl);
}

template<typename ImplType>
TensorImpl& TensorImpl::operator+=(const ImplType& exp_impl) {
    CHECK_EXP_BROADCAST(*this, exp_impl);
    storage_.increment_version();
    if(is_contiguous())
        return __inplacement_add(exp_impl);
    else
        return __inplacement_add_uncontiguous(exp_impl);
}

template<typename ImplType>
TensorImpl& TensorImpl::__assign(const ImplType& exp_impl) {
    IndexArray inds(ndim());
    for(index_t i = 0; i < shape_.dsize(); ++i) {
        for(index_t ii = i, j = 0; j < ndim(); ++j) {
            if(stride_[j] != 0) {
                inds[j] = ii / stride_[j];
                ii %= stride_[j];
            } else {
                inds[j] = 0;
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
            } else {
                inds[j] = 0;
            }
        }
        storage_[i] += exp_impl.eval(inds);
    }
    return *this;
}

template<typename ImplType>
TensorImpl& TensorImpl::__assign_uncontiguous(const ImplType& exp_impl) {
    IndexArray inds(ndim());
    IndexArray cur(ndim());
    index_t idx = 0;
    cur.memset(0);

    while(true) {
        if(idx == ndim()) {
            --idx;
            index_t offset = 0;
            for(index_t i = 0; i < inds.size(); ++i)
                offset += stride_[i] * inds[i];
            storage_[offset] = exp_impl.eval(inds);
        } else {
            while(idx < ndim() && cur[idx] == size(idx)) {
                cur[idx] = 0;
                --idx;
            }
            if(idx > ndim()) break;

            inds[idx] = cur[idx];
            ++cur[idx];
            ++idx;
        }
    }
    return *this;
}

template<typename ImplType>
TensorImpl& TensorImpl::__inplacement_add_uncontiguous(const ImplType& exp_impl) {
    IndexArray inds(ndim());
    IndexArray cur(ndim());
    index_t idx = 0;
    cur.memset(0);

    while(true) {
        if(idx == ndim()) {
            --idx;
            index_t offset = 0;
            for(index_t i = 0; i < inds.size(); ++i)
                offset += stride_[i] * inds[i];
            storage_[offset] += exp_impl.eval(inds);
        } else {
            while(idx < ndim() && cur[idx] == size(idx)) {
                cur[idx] = 0;
                --idx;
            }
            if(idx > ndim()) break;

            inds[idx] = cur[idx];
            ++cur[idx];
            ++idx;
        }
    }
    return *this;
}

template<typename ImplType>
void TensorImpl::backward(const ImplType& grad) {
    gradmeta_ptr_->grad_ += grad;
    if(gradcount() == 0) {
        auto& grad_fn = *(gradmeta_ptr_->grad_fn_ptr_);
        grad_fn(gradmeta_ptr_->grad_);
    }
}

inline void TensorImpl::backward(void) {
    if(gradcount() == 0) {
        auto& grad_fn = *(gradmeta_ptr_->grad_fn_ptr_);
        grad_fn();
    }
}

}  // namespace st
#endif