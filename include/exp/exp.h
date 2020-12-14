#ifndef EXP_EXP_H
#define EXP_EXP_H

#include "exp/exp_impl.h"
#include "exp/operator/basic_op.h"

namespace st {
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
template<typename ImplType> 
struct Exp {
public:
    explicit Exp(Alloc::NontrivialUniquePtr<ImplType>&& ptr)
            :impl_ptr_(std::move(ptr), false) {}
    const ExpImplPtr<ImplType>& impl_ptr(void) const { return impl_ptr_; }
    const ImplType& impl(void) const { return *impl_ptr_; }

protected:
    ExpImplPtr<ImplType> impl_ptr_;
};

// forward declaration and using declaration
namespace op {
template<typename OIType>
Exp<UnaryExpImpl<Minus, OIType>> 
operator-(const Exp<OIType>& operand);

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, RhsImplType>>
operator+(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs);

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, RhsImplType>>
operator*(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs);

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, RhsImplType>>
operator-(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs);
}  // namespace op

using op::operator+;
using op::operator*;
using op::operator-;

}  // namespace st

#endif