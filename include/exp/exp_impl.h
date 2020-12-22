#ifndef EXP_EXP_IMPL_H
#define EXP_EXP_IMPL_H

#include <memory>
#include <initializer_list>

#include "utils/allocator.h"
#include "utils/base_config.h"
#include "utils/array.h"

#include "exp/grad_impl.h"
#include "exp/operator/log_softmax.h"
#include "exp/operator/nll_loss.h"
#include "exp/operator/reduce_op.h"
#include "exp/operator/conv.h"
#include "exp/operator/constant.h"

namespace st {

// forward declaration
template<typename T> class ExpImpl;
template<typename T> class ExpImplPtr;
template<typename T> using OperandImplPtr = ExpImplPtr<T>;
template<typename Op, typename OIType> class UnaryExpImpl;
template<typename Op, typename LhsImplType, typename RhsImplType> class BinaryExpImpl;


template<typename ImplType>
class ExpImpl {
public:
    index_t refcount(void) const { return refcount_; }
    index_t gradcount(void) const { return gradcount_; }
    friend class ExpImplPtr<ImplType>;
private:
    index_t refcount_ = 0;
    index_t gradcount_ = 0;
};

template<typename ImplType> 
class ExpImplPtr {
public:
    ExpImplPtr(Alloc::NontrivialUniquePtr<ImplType>&& ptr, bool with_grad)
            : ptr_(ptr.release()), 
              with_grad_(with_grad && static_cast<ImplType*>(ptr_)->requires_grad()) {
        increment_counters();
    }
    ExpImplPtr(const ImplType& impl, bool with_grad)
            : ptr_(const_cast<ImplType*>(&impl)), 
              with_grad_(with_grad && static_cast<ImplType*>(ptr_)->requires_grad()) { 
        increment_counters();
    }
    ExpImplPtr(const ExpImplPtr& other, bool with_grad)
            : ptr_(other.ptr_),
              with_grad_(with_grad && static_cast<ImplType*>(ptr_)->requires_grad()) {
        increment_counters();
    }
    ~ExpImplPtr() { decrease_refcount(); }

    ImplType* operator->(void) const { return static_cast<ImplType*>(ptr_); }
    const ImplType& operator*(void) const { return *static_cast<ImplType*>(ptr_); }
    explicit operator bool() const { return ptr_ != nullptr; }

    template<typename GradImplType>
    void invoke_backward(const GradImplType& grad) {
        auto ptr = static_cast<ImplType*>(ptr_);
        if(ptr->requires_grad()) {
            if(with_grad_)
                -- ptr_->gradcount_;
            ptr->backward(grad);
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

    ExpImpl<ImplType>* ptr_;
    bool with_grad_;
    Alloc::nontrivial_delete_handler<ImplType> delete_handler;
};
}  // namespace st


namespace st {
template<typename Op, typename OIType>  // OIType = OperandImplType
class UnaryExpImpl 
        : public ExpImpl<UnaryExpImpl<Op, OIType>> {
public:
    explicit UnaryExpImpl(const OperandImplPtr<OIType>& ptr)
            : operand_ptr_(ptr, true) {}

    index_t ndim(void) const { return Op::ndim(*operand_ptr_); }
    index_t size(index_t idx) const { return Op::size(idx, *operand_ptr_); }

    data_t eval(IndexArray& inds) const {
        return Op::map(inds, *operand_ptr_);
    }

   IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    bool requires_grad(void) const { return operand_ptr_->requires_grad(); }

    template<typename GIType>
    void backward(const GIType& grad) {
        CHECK_EQUAL(this->gradcount(), 0, "Reused ExpImpl can't be backward.");

        UnaryGradImpl<typename Op::Grad, GIType, OIType> out_grad(
            grad, *operand_ptr_
        );
        operand_ptr_.invoke_backward(out_grad);
    }
private:
    OperandImplPtr<OIType> operand_ptr_;
};

template<typename Op, typename LhsImplType, typename RhsImplType>
class BinaryExpImpl
        : public ExpImpl<BinaryExpImpl<Op, LhsImplType, RhsImplType>> {
public:
    BinaryExpImpl(const OperandImplPtr<LhsImplType>& lhs_ptr,
                  const OperandImplPtr<RhsImplType>& rhs_ptr)
            : lhs_ptr_(lhs_ptr, true),
              rhs_ptr_(rhs_ptr, true) {}

    index_t ndim(void) const { return Op::ndim(*lhs_ptr_, *rhs_ptr_); }
    index_t size(index_t idx) const { return Op::size(idx, *lhs_ptr_, *rhs_ptr_); }

    data_t eval(IndexArray& inds) const {
        return Op::map(inds, *lhs_ptr_, *rhs_ptr_);
    }

    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    bool requires_grad(void) const { 
        return lhs_ptr_->requires_grad() || rhs_ptr_->requires_grad(); 
    }

    template<typename GIType>
    void backward(const GIType& grad) {
        CHECK_EQUAL(this->gradcount(), 0, "Reused ExpImpl can't be backward.");

        BinaryGradImpl<typename Op::LhsGrad, GIType, LhsImplType, RhsImplType> 
        lhs_grad(grad, *lhs_ptr_, *rhs_ptr_);
        lhs_ptr_.invoke_backward(lhs_grad);
        
        BinaryGradImpl<typename Op::RhsGrad, GIType, LhsImplType, RhsImplType> 
        rhs_grad(grad, *lhs_ptr_, *rhs_ptr_);
        rhs_ptr_.invoke_backward(rhs_grad);
    }

private:
    OperandImplPtr<LhsImplType> lhs_ptr_;
    OperandImplPtr<RhsImplType> rhs_ptr_;
};
} // namespace st


// Template specialization for ExpImplPtr, UnaryExpImpl and BinaryExpImpl.
// 1. ExpImplPtr need be specialized for TensorImpl,
//    because TensorImpl' version need check before invoking its backward.
//    But to make the dependency between header files more clear, I put this
//    specialization in tensor/tensor_impl.h
// 2. Some operators need store some states or parameters in UnaryExpImpl,
//    or BinaryExpImpl.
namespace st {

template<typename OIType>
class UnaryExpImpl<op::LogSoftmax, OIType>
        : public ExpImpl<UnaryExpImpl<op::LogSoftmax, OIType>> {
public:
    explicit UnaryExpImpl(const OperandImplPtr<OIType>& ptr)
            : operand_ptr_(ptr, true),
              n_batch_(operand_ptr_->size(0)),
              batch_sum_exp_(
                  Alloc::unique_allocate<data_t>(
                      sizeof(data_t) * operand_ptr_->size(0))),
              batch_max_cls_(
                  Alloc::unique_allocate<data_t>(
                      sizeof(data_t) * operand_ptr_->size(0))) {
        op::LogSoftmax::precompute(*operand_ptr_, batch_sum_exp_.get(), 
                                   batch_max_cls_.get());
    }

    index_t ndim(void) const { return op::LogSoftmax::ndim(*operand_ptr_); }
    index_t size(index_t idx) const { return op::LogSoftmax::size(idx, *operand_ptr_); }
    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    data_t eval(IndexArray& inds) const {
        return op::LogSoftmax::map(inds, *operand_ptr_, 
                                   batch_sum_exp_.get(), batch_max_cls_.get());
    }

    bool requires_grad(void) const { return operand_ptr_->requires_grad(); }

    template<typename GIType>
    void backward(const GIType& grad) {
        CHECK_EQUAL(this->gradcount(), 0, "Reused ExpImpl can't be backward.");

        UnaryGradImpl<typename op::LogSoftmax::Grad, GIType, OIType> out_grad(
            grad, *operand_ptr_, batch_sum_exp_.get(), batch_max_cls_.get()
        );
        operand_ptr_.invoke_backward(out_grad);
    }
private:
    OperandImplPtr<OIType> operand_ptr_;
    index_t n_batch_;
    Alloc::TrivialUniquePtr<data_t> batch_sum_exp_;
    Alloc::TrivialUniquePtr<data_t> batch_max_cls_;
};

template<typename OIType>
class UnaryExpImpl<op::Mean, OIType>
        : public ExpImpl<UnaryExpImpl<op::Mean, OIType>> {
public:
    explicit UnaryExpImpl(const OperandImplPtr<OIType>& ptr, index_t reduce_dim)
            : operand_ptr_(ptr, true),
              reduce_dim_(reduce_dim) {}

    index_t ndim(void) const { return op::Mean::ndim(*operand_ptr_); }
    index_t size(index_t idx) const { 
        return op::Mean::size(idx, *operand_ptr_, reduce_dim_); 
    }
    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }


    data_t eval(IndexArray& inds) const {
        return op::Mean::map(inds, *operand_ptr_, reduce_dim_);
    }

    bool requires_grad(void) const { return operand_ptr_->requires_grad(); }

    template<typename GIType>
    void backward(const GIType& grad) {
        CHECK_EQUAL(this->gradcount(), 0, "Reused ExpImpl can't be backward.");

        UnaryGradImpl<typename op::Mean::Grad, GIType, OIType> out_grad(
            grad, *operand_ptr_, reduce_dim_
        );
        operand_ptr_.invoke_backward(out_grad);
    }

private:
    OperandImplPtr<OIType> operand_ptr_;
    index_t reduce_dim_;    
};

template<typename OIType>
class UnaryExpImpl<op::Max, OIType>
        : public ExpImpl<UnaryExpImpl<op::Max, OIType>> {
public:
    explicit UnaryExpImpl(const OperandImplPtr<OIType>& ptr, index_t reduce_dim)
            : operand_ptr_(ptr, true),
              reduce_dim_(reduce_dim) {}

    index_t ndim(void) const { return op::Max::ndim(*operand_ptr_); }
    index_t size(index_t idx) const { 
        return op::Max::size(idx, *operand_ptr_, reduce_dim_); 
    }
    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }


    data_t eval(IndexArray& inds) const {
        return op::Max::map(inds, *operand_ptr_, reduce_dim_);
    }

    bool requires_grad(void) const { return operand_ptr_->requires_grad(); }

    template<typename GIType>
    void backward(const GIType& grad) {
        CHECK_EQUAL(this->gradcount(), 0, "Reused ExpImpl can't be backward.");

        UnaryGradImpl<typename op::Max::Grad, GIType, OIType> out_grad(
            grad, *operand_ptr_, reduce_dim_
        );
        operand_ptr_.invoke_backward(out_grad);
    }

private:
    OperandImplPtr<OIType> operand_ptr_;
    index_t reduce_dim_;    
};

template<typename OIType>
class UnaryExpImpl<op::Argmax, OIType>
        : public ExpImpl<UnaryExpImpl<op::Argmax, OIType>> {
public:
    explicit UnaryExpImpl(const OperandImplPtr<OIType>& ptr, index_t reduce_dim)
            : operand_ptr_(ptr, true),
              reduce_dim_(reduce_dim) {}

    index_t ndim(void) const { return op::Argmax::ndim(*operand_ptr_); }
    index_t size(index_t idx) const { 
        return op::Argmax::size(idx, *operand_ptr_, reduce_dim_); 
    }
    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    index_t eval(IndexArray& inds) const {
        return op::Argmax::map(inds, *operand_ptr_, reduce_dim_);
    }

    bool requires_grad(void) const { return operand_ptr_->requires_grad(); }

    template<typename GIType>
    void backward(const GIType& grad) const {
        THROW_ERROR("NotImplementError for backward of Argmax");
    }

private:
    OperandImplPtr<OIType> operand_ptr_;
    index_t reduce_dim_;    
};

template<typename OIType>
class UnaryExpImpl<op::NLLLoss, OIType>
        : public ExpImpl<UnaryExpImpl<op::NLLLoss, OIType>> {
public:
    explicit UnaryExpImpl(const OperandImplPtr<OIType>& ptr,
                          const std::shared_ptr<index_t>& batch_label)
            : operand_ptr_(ptr, true),
              batch_label_(batch_label) {}

    index_t ndim(void) const { return op::NLLLoss::ndim(*operand_ptr_); }
    index_t size(index_t idx) const { return op::NLLLoss::size(idx, *operand_ptr_); }
    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    data_t eval(IndexArray& inds) const {
        return op::NLLLoss::map(inds, *operand_ptr_, batch_label_.get());
    }

    bool requires_grad(void) const { return operand_ptr_->requires_grad(); }

    template<typename GIType>
    void backward(const GIType& grad) {
        CHECK_EQUAL(this->gradcount(), 0, "Reused ExpImpl can't be backward.");

        UnaryGradImpl<typename op::NLLLoss::Grad, GIType, OIType> out_grad(
            grad, *operand_ptr_, batch_label_.get()
        );
        operand_ptr_.invoke_backward(out_grad);
    }
private:
    OperandImplPtr<OIType> operand_ptr_;
    std::shared_ptr<index_t> batch_label_;  
};

template<typename OIType>
class UnaryExpImpl<op::Img2col, OIType>
        : public ExpImpl<UnaryExpImpl<op::Img2col, OIType>> {
public:
    UnaryExpImpl(const OperandImplPtr<OIType>& ptr,
                 const op::Img2col::Wsize& kernel_size,
                 const op::Img2col::Wsize& stride_size,
                 const op::Img2col::Wsize& padding_size) 
            : operand_ptr_(ptr, true),
              kernel_size_(kernel_size),
              stride_size_(stride_size),
              padding_size_(padding_size) {
        index_t b = operand_ptr_->size(0);
        index_t c = operand_ptr_->size(1);
        index_t h = operand_ptr_->size(2);
        index_t w = operand_ptr_->size(3);
        out_size_.first = 
            (h + 2*padding_size_.first - kernel_size_.first) / stride_size_.first + 1;
        out_size_.second = 
            (w + 2*padding_size_.second - kernel_size_.second) / stride_size_.second + 1;
        shape_.first = out_size_.first * out_size_.second * b;
        shape_.second = c * kernel_size_.first * kernel_size_.second;
    }

    index_t ndim(void) const { return op::Img2col::ndim(*operand_ptr_); }
    index_t size(index_t idx) const {
        return op::Img2col::size(idx, *operand_ptr_, shape_);
    }
    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }
    const op::Img2col::Wsize& conv_feat_size(void) const { return out_size_; }

    data_t eval(IndexArray& inds) const {
        return op::Img2col::map(inds, *operand_ptr_, kernel_size_,
                                  stride_size_, padding_size_, out_size_);
    }

    bool requires_grad(void) const { return operand_ptr_->requires_grad(); }

    template<typename GIType>
    void backward(const GIType& grad) {
        CHECK_EQUAL(this->gradcount(), 0, "Reused ExpImpl can't be backward.");

        UnaryGradImpl<typename op::Img2col::Grad, GIType, OIType> out_grad(
            grad, *operand_ptr_, kernel_size_, stride_size_,
            padding_size_, out_size_
        );
        operand_ptr_.invoke_backward(out_grad);
    }
private:
    OperandImplPtr<OIType> operand_ptr_;
    op::Img2col::Wsize kernel_size_;
    op::Img2col::Wsize stride_size_;
    op::Img2col::Wsize padding_size_;
    op::Img2col::Wsize out_size_;
    op::Img2col::Wsize shape_;
};

template<typename OIType>
class UnaryExpImpl<op::MaxPool2d, OIType>
        : public ExpImpl<UnaryExpImpl<op::MaxPool2d, OIType>> {
public:
    UnaryExpImpl(const OperandImplPtr<OIType>& ptr,
                 const op::MaxPool2d::Wsize& kernel_size,
                 const op::MaxPool2d::Wsize& stride_size,
                 const op::MaxPool2d::Wsize& padding_size) 
            : operand_ptr_(ptr, true),
              kernel_size_(kernel_size),
              stride_size_(stride_size),
              padding_size_(padding_size) {
        index_t h = operand_ptr_->size(2);
        index_t w = operand_ptr_->size(3);
        out_size_.first = 
            (h + 2*padding_size_.first - kernel_size_.first) / stride_size_.first + 1;
        out_size_.second = 
            (w + 2*padding_size_.second - kernel_size_.second) / stride_size_.second + 1;
    }

    index_t ndim(void) const { return op::MaxPool2d::ndim(*operand_ptr_); }
    index_t size(index_t idx) const {
        return op::MaxPool2d::size(idx, *operand_ptr_, out_size_);
    }
    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    data_t eval(IndexArray& inds) const {
        return op::MaxPool2d::map(inds, *operand_ptr_, kernel_size_,
                                  stride_size_, padding_size_);
    }

    bool requires_grad(void) const { return operand_ptr_->requires_grad(); }

    template<typename GIType>
    void backward(const GIType& grad) {
        THROW_ERROR("NotImplmentError in backward of MaxPool2d");
    }
private:
    OperandImplPtr<OIType> operand_ptr_;
    op::MaxPool2d::Wsize kernel_size_;
    op::MaxPool2d::Wsize stride_size_;
    op::MaxPool2d::Wsize padding_size_;
    op::MaxPool2d::Wsize out_size_;
};

template<>
class UnaryExpImpl<op::Constant, data_t>
        : public ExpImpl<UnaryExpImpl<op::Constant, data_t>> {
public:
    UnaryExpImpl(data_t value, IndexArray&& size) 
            : value_(value), shape_(std::move(size)) {}

    index_t ndim(void) const { return shape_.size(); }
    index_t size(index_t idx) const { return shape_[idx]; }
    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    data_t eval(IndexArray& inds) const {
        return op::Constant::map(inds, value_);
    }

    bool requires_grad(void) const { return false; }

    template<typename GIType>
    void backward(const GIType& grad) {
        THROW_ERROR("NotImplementError in backward of Constant");
    }

private:
    data_t value_;
    IndexArray shape_;
};

}  // namespace st
#endif