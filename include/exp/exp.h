#ifndef EXP_EXP_H
#define EXP_EXP_H

#include <memory>
#include <utility>

#include "utils/allocator.h"
#include "utils/base_config.h"

namespace st {

// forward declaration
template<typename T> class ExpImplPtr;
template<typename T> class ExpImpl;

/*
struct Exp is a bit like the shell of class ExpImpl.
The meaning of this shell is to keep ExpImpl alive in heap memory instead of stack 
memory. Think about this case:
    Exp compute(Exp& a, Exp& b) {
        Exp mul_exp = a * b;
        Exp add_exp = a + mul_exp;
        return add_exp;
    } 
    int main() {
        ...
        Tensor result = compute(a, b);
        return 0;
    }
We implement Expression Template, so we expect lazy computation. In this case, we 
expect the computation is done in Tensor::operator=(const Exp&). But at this time,
mul_exp has been deconstructed since compute(a, b) returned. So Exp need a shell.
*/
template<typename Subtype> 
struct Exp {
    ExpImplPtr<Subtype> impl_ptr_;
    Exp(Alloc::NontrivialUniquePtr<Subtype>&& ptr)
            :impl_ptr_(std::move(ptr)) {}
};

template<typename Subtype> 
class ExpImpl {
public:
    const Subtype& self() const {
        return *static_cast<const Subtype*>(this);
    }

    friend class ExpImplPtr<Subtype>;
private:
    index_t refcount_ = 0;
    index_t gradcount_ = 0;
};

template<typename Subtype> 
class ExpImplPtr {
public:
    using Impl = ExpImpl<Subtype>;

    ExpImplPtr(Alloc::NontrivialUniquePtr<Subtype>&& ptr)
            : ptr_(ptr.release()) { increment_refcount(); }
    ~ExpImplPtr() { decrease_refcount(); }

    Subtype* operator->(void) const { return static_cast<Subtype*>(ptr_); }
    const Subtype& operator*(void) const { return ptr_->self(); }
private:
    void increment_refcount() { ++ptr_->refcount_; }

    void decrease_refcount() {
        --ptr_->refcount_;
        if(ptr_->refcount_ == 0)
            delete_handler(static_cast<void*>(ptr_));
    }

    Impl* ptr_;
    Alloc::nontrivial_delete_handler<Subtype> delete_handler;
};

} // namespace st end

#endif