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

namespace st {

// forward declaration
template<typename T> class ExpImpl;
template<typename T> class ExpImplPtr;
template<typename T> using OperandImplPtr = ExpImplPtr<T>;
template<typename Op, typename OIType> class UnaryExpImpl;
template<typename Op, typename LhsImplType, typename RhsImplType> class BinaryExpImpl;


template<typename Subtype>
class ExpImpl {
public:
    friend class ExpImplPtr<Subtype>;
private:
    index_t refcount_ = 0;
    index_t gradcount_ = 0;
};

template<typename Subtype> 
class ExpImplPtr {
public:
    explicit ExpImplPtr(Alloc::NontrivialUniquePtr<Subtype>&& ptr)
            : ptr_(ptr.release()) { increment_refcount(); }
    ExpImplPtr(const ExpImplPtr& other)
            : ptr_(other.ptr_) { increment_refcount(); }
    ~ExpImplPtr() { decrease_refcount(); }

    Subtype* operator->(void) const { return static_cast<Subtype*>(ptr_); }
    const Subtype& operator*(void) const { return *static_cast<Subtype*>(ptr_); }
private:
    void increment_refcount() { ++ptr_->refcount_; }

    void decrease_refcount() {
        --ptr_->refcount_;
        if(ptr_->refcount_ == 0)
            delete_handler(static_cast<void*>(ptr_));
    }

    ExpImpl<Subtype>* ptr_;
    Alloc::nontrivial_delete_handler<Subtype> delete_handler;
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

// Some operators need store some states to accelerate computation.
// We need partitially specialize the UnaryExpImpl or BinaryExpImpl here.
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

}  // namespace st
#endif