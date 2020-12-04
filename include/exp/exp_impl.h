#ifndef EXP_EXP_IMPL_H
#define EXP_EXP_IMPL_H

#include <memory>

#include "utils/allocator.h"
#include "utils/base_config.h"
#include "utils/array.h"
#include "exp/operator/basic_op.h"

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

    data_t eval(index_t idx) const { 
        return Op::map(idx, *operand_ptr_);
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

    data_t eval(index_t idx) const {
        return Op::map(idx, *lhs_ptr_, *rhs_ptr_);
    }

private:
    OperandImplPtr<LhsImplType> lhs_ptr_;
    OperandImplPtr<RhsImplType> rhs_ptr_;
};

} // namespace st end

#endif