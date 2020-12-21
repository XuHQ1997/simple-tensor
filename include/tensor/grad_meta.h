#ifndef TENSOR_GRAD_META_H
#define TENSOR_GRAD_META_H

#include "utils/exception.h"
#include "exp/grad_impl.h"
#include "tensor/tensor_impl.h"


namespace st {
// TensorImpl isn't a template class, which means AutoGradMeta couldn't be a template
// class. But AutoGradMeta need store the information of next_exp via template class
// ExpImplPtr<ImplType>. 
// 
// We need dynamic polymorphism to implement this.
struct GradFn {
    virtual void operator()(void) = 0;
    virtual void operator()(const Storage& grad, const Shape& shape,
                            const IndexArray& stride) = 0;
    virtual ~GradFn() = default;

    struct TensorGradImpl: public GradImpl<TensorGradImpl> {
        const Storage& storage_;
        const Shape& shape_;
        const IndexArray& stride_;

        TensorGradImpl(const Storage& storage, const Shape& shape, 
                       const IndexArray& stride)
                : storage_(storage),
                  shape_(shape),
                  stride_(stride) {}

        data_t eval(IndexArray& inds) const {
            index_t offset = 0;
            for(int i = 0; i < shape_.ndim(); ++i)
                offset += inds[i] * stride_[i];
            return storage_[offset];
        }
    };
};

template<typename ImplType> 
class __GradFn: public GradFn {
public:
    __GradFn(const ImplType& impl) : next_exp_(impl, false) {}
    ~__GradFn() = default;

    void operator()(void) override { 
        THROW_ERROR("Need grad when invoke backward method of a expression.");
    }

    void operator()(const Storage& grad, const Shape& shape, 
                    const IndexArray& stride) override {
        TensorGradImpl grad_exp_impl(grad, shape, stride);
        next_exp_.invoke_backward(grad_exp_impl);
    }
private:
    ExpImplPtr<ImplType> next_exp_;
};

template<>
struct __GradFn<TensorImpl>: public GradFn {
public:
    __GradFn(const TensorImpl& impl) : next_exp_(impl, false) {}
    ~__GradFn() = default;

    void operator()(void) override {
        next_exp_.invoke_backward();
    }

    void operator()(const Storage& grad, const Shape& shape,
                    const IndexArray& stride) override {
        TensorGradImpl grad_exp_impl(grad, shape, stride);
        next_exp_.invoke_backward(grad_exp_impl);
    }
private:
    ExpImplPtr<TensorImpl> next_exp_;
};


struct AutoGradMeta {

    Storage grad_;
    bool from_view_;
    std::shared_ptr<GradFn> grad_fn_ptr_;

    AutoGradMeta(const Shape& tensor_shape)
            : grad_(tensor_shape.dsize(), 0),
              from_view_(false),
              grad_fn_ptr_(nullptr) {}
    
    AutoGradMeta(const Storage& grad, index_t offset)
            : grad_(grad, offset),
              from_view_(false),
              grad_fn_ptr_(nullptr) {}

    void set_from_view(bool from_view) { from_view_ = from_view; }

    template<typename ImplType>
    void set_grad_fn(const ImplType& impl) {
        auto ptr = Alloc::shared_construct<__GradFn<ImplType>>(impl);
        grad_fn_ptr_ = ptr;
    }
};
}  // namespace st
#endif


/*
Backward workflow:
1. Tensor::backward()
    1. check requires_grad of TensorImpl;
    2. check TensorImpl is scalar tensor;

2. ExpImplPtr<Impl>::invoke_backward()
    1. ExpImpl<TensorImpl>::invoke_backward()
        1. check with_grad;
        2. decrease gradcount of TensorImpl;
        3. check version
    2. ExpImpl<Impl>::invoke_backward()
        1. check with_grad
        2. decrease gradcount of Impl

3. ExpImpl<Impl>::backward()
    1. UnaryExpImpl
        1. check gradcount == 0. if not, throw Error
    2. BinaryExpImpl
        1. check gradcount == 0. if not, throw Error
    3. TensorImpl
        1. acculate grad
        2. check gradcount == 0. if so, call grad_fn
        3. check from_view is False. if so, don't pass grad
*/