#ifndef TENSOR_GRAD_META_H
#define TENSOR_GRAD_META_H

#include "utils/exception.h"
#include "exp/exp_impl.h"
#include "exp/function.h"
#include "tensor/tensor_impl.h"


namespace st {
// TensorImpl isn't a template class, which means AutoGradMeta couldn't be a template
// class. But AutoGradMeta need store the information of next_exp via template class
// ExpImplPtr<ImplType>. 
// 
// We need dynamic polymorphism to implement this.
struct GradFn {
    virtual void operator()(void) = 0;
    virtual void operator()(TensorImpl& grad) = 0;
};

template<typename ImplType> 
class __GradFn: public GradFn {
public:
    __GradFn(const ImplType& impl) : next_exp_(impl, false) {}

    void operator()(void) override { 
        THROW_ERROR("Need grad when invoke backward method of a expression.");
    }

    void operator()(TensorImpl& grad) override { 
        next_exp_.invoke_backward(grad);
    }
private:
    ExpImplPtr<ImplType> next_exp_;
};

template<>
struct __GradFn<TensorImpl>: public GradFn {
public:
    __GradFn(const TensorImpl& impl) : next_exp_(impl, false) {}

    virtual void operator()(void) override {
        next_exp_.invoke_backward();
    }

    virtual void operator()(TensorImpl& grad) override {
        next_exp_.invoke_backward(grad);
    }
private:
    ExpImplPtr<TensorImpl> next_exp_;
};


struct AutoGradMeta {

    TensorImpl grad_;
    bool from_view_;
    Alloc::NontrivialUniquePtr<GradFn> grad_fn_ptr_;

    template<typename... Args>
    AutoGradMeta(Args... args) 
            : grad_(std::forward<Args...>(args...), /*requires_grad=*/false),
              from_view_(false),
              grad_fn_ptr_() {}

    void set_from_view(bool from_view) { from_view_ = from_view_; }

    template<typename ImplType>
    void set_grad_fn(const ImplType& impl) {
        grad_fn_ptr_ = Alloc::unique_construct<__GradFn>(impl);
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