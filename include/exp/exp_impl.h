#ifndef EXP_EXP_IMPL_H
#define EXP_EXP_IMPL_H

#include <memory>
#include <initializer_list>

#include "utils/allocator.h"
#include "utils/base_config.h"
#include "utils/array.h"

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
    ExpImplPtr() : ptr_(nullptr) {}
    explicit ExpImplPtr(Alloc::NontrivialUniquePtr<ImplType>&& ptr)
            : ptr_(ptr.release()) { increment_refcount(); }
    ExpImplPtr(const ExpImplPtr& other)
            : ptr_(other.ptr_) { increment_refcount(); }
    ~ExpImplPtr() { decrease_refcount(); }

    ImplType* operator->(void) const { return static_cast<ImplType*>(ptr_); }
    const ImplType& operator*(void) const { return *static_cast<ImplType*>(ptr_); }
    explicit operator bool() const { return ptr_ != nullptr; }

    template<typename GradImplType>
    void invoke_backward(const ExpImpl<GradImplType>& grad) {
        static_cast<ImplType*>(ptr_)->backward(grad);
    }
private:
    void increment_refcount() { ++ptr_->refcount_; }

    void decrease_refcount() {
        --ptr_->refcount_;
        if(ptr_->refcount_ == 0)
            delete_handler(static_cast<void*>(ptr_));
    }

    ExpImpl<ImplType>* ptr_;
    Alloc::nontrivial_delete_handler<ImplType> delete_handler;
};

template<typename Op, typename OIType>  // OIType = OperandImplType
class UnaryExpImpl : public ExpImpl<UnaryExpImpl<Op, OIType>> {
public:
    explicit UnaryExpImpl(const OperandImplPtr<OIType>& ptr)
            : operand_ptr_(ptr) {}
    
    index_t ndim(void) const { return Op::ndim(*operand_ptr_); }
    index_t size(index_t idx) const { return Op::size(idx, *operand_ptr_); }

    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    data_t eval(IndexArray& inds) const {
        return Op::map(inds, *operand_ptr_);
    }

private:
    OperandImplPtr<OIType> operand_ptr_;
};

template<typename Op, typename LhsImplType, typename RhsImplType>
class BinaryExpImpl : public ExpImpl<BinaryExpImpl<Op, LhsImplType, RhsImplType>> {
public:
    BinaryExpImpl(const OperandImplPtr<LhsImplType>& lhs_ptr,
                  const OperandImplPtr<RhsImplType>& rhs_ptr)
            : lhs_ptr_(lhs_ptr), 
              rhs_ptr_(rhs_ptr) {}

    index_t ndim(void) const { return Op::ndim(*lhs_ptr_, *rhs_ptr_); }
    index_t size(index_t idx) const { return Op::size(idx, *lhs_ptr_, *rhs_ptr_); }

    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    data_t eval(IndexArray& inds) const {
        return Op::map(inds, *lhs_ptr_, *rhs_ptr_);
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
            : operand_ptr_(ptr),
              batch_sum_exp_(Alloc::shared_allocate<data_t>(operand_ptr_->size(0))),
              batch_max_cls_(Alloc::shared_allocate<data_t>(operand_ptr_->size(0))) {
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

private:
    OperandImplPtr<OIType> operand_ptr_;
    
    std::shared_ptr<data_t> batch_sum_exp_;
    std::shared_ptr<data_t> batch_max_cls_;
};

template<typename OIType>
class UnaryExpImpl<op::MeanReduce, OIType>
        : public ExpImpl<UnaryExpImpl<op::MeanReduce, OIType>> {
public:
    explicit UnaryExpImpl(const OperandImplPtr<OIType>& ptr, index_t reduce_dim)
            : operand_ptr_(ptr),
              reduce_dim_(reduce_dim) {}

    index_t ndim(void) const { return op::MeanReduce::ndim(*operand_ptr_); }
    index_t size(index_t idx) const { 
        return op::MeanReduce::size(idx, *operand_ptr_, reduce_dim_); 
    }

    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    data_t eval(IndexArray& inds) const {
        return op::MeanReduce::map(inds, *operand_ptr_, reduce_dim_);
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
            : operand_ptr_(ptr),
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
            : operand_ptr_(ptr),
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
            : operand_ptr_(ptr),
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
        shape_.first = c * kernel_size_.first * kernel_size_.second;
        shape_.second = out_size_.first * out_size_.second * b;
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

    data_t eval(IndexArray& inds) const {
        return op::Img2col::map(inds, *operand_ptr_, kernel_size_,
                                  stride_size_, padding_size_, out_size_);
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
            : operand_ptr_(ptr),
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

private:
    OperandImplPtr<OIType> operand_ptr_;

    op::MaxPool2d::Wsize kernel_size_;
    op::MaxPool2d::Wsize stride_size_;
    op::MaxPool2d::Wsize padding_size_;
    op::MaxPool2d::Wsize out_size_;
};

template<> class UnaryExpImpl<op::Constant, data_t>
        : public ExpImpl<UnaryExpImpl<op::Constant, data_t>> {
public:
    UnaryExpImpl(data_t value) : value_(value) {}
    index_t ndim(void) const { return op::Constant::ndim(); }
    index_t size(index_t idx) const { return op::Constant::size(idx); }

    IndexArray size(void) const {
        IndexArray shape(ndim());
        for(index_t i = 0; i < shape.size(); ++i)
            shape[i] = size(i);
        return shape;
    }

    data_t eval(IndexArray& inds) const {
        return op::Constant::map(inds, value_);
    }
private:
    data_t value_;
};

}  // namespace st
#endif