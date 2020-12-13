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
// We need dynamic polymorphism, virtual function, to implement this.
struct AutoGradMeta {
public:

    struct GradFn {
        virtual void operator()(void) = 0;
        virtual void operator()(TensorImpl& grad) = 0;
    };

    template<typename... Args>
    AutoGradMeta(Args... args) 
            : grad_(std::forward(args...), /*requires_grad=*/false) {}

    TensorImpl& grad() { return grad_; }
    bool from_view() { return from_view_; }
    GradFn& grad_fn() { return *grad_fn_ptr_; }

    void zero_grad(void) { grad_ = op::constant(0); }

    void set_from_view(bool from_view) { from_view_ = from_view_; }

    template<typename ImplType>
    void set_grad_fn(const ImplType& impl) {
        grad_fn_ptr_ = Alloc::unique_construct<__GradFn>(impl);
    }

private:
    TensorImpl grad_;
    bool from_view_;
    Alloc::NontrivialUniquePtr<GradFn> grad_fn_ptr_;
};

template<typename ImplType> 
class __GradFn: public AutoGradMeta::GradFn {
public:
    __GradFn(const Impl& impl) : next_exp_(impl) {}

    void operator()(void) override { 
        THROW_ERROR("Need grad when invoke backward method of a expression.");
    }

    void operator()(TensorImpl& grad) override { 
        next_exp_.invoke_backward(grad);
    }
private:
    ExpImplPtr<ImplType> next_exp_;
};

template<> struct __GradFn<TensorImpl>: public AutoGradMeta::GradFn {
public:
    __GradFn(const TensorImpl& impl) : next_exp_(impl) {}

    virtual void operator()(void) override {
        next_exp_.invoke_backward();
    }

    virtual void operator()(TensorImpl& grad) override {
        next_exp_.invoke_backward(grad);
    }
private:
    ExpImplPtr<TensorImpl> next_exp_;
};

}  // namespace st
#endif