#ifndef EXP_FUNCTION_H
#define EXP_FUNCTION_H

#include "exp/operator/basic_op.h"
#include "exp/exp_impl.h"
#include "utils/exception.h"

namespace st {

template<typename ImplType> class Exp;

namespace op {

namespace {
template<typename Op, typename OIType>  // OIType = OperandImplType
Exp<UnaryExpImpl<Op, OIType>> 
unary_operation_function(const Exp<OIType>& operand) {
    return Exp<UnaryExpImpl<Op, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<Op, OIType>>(
            operand.impl_ptr()
        )
    );
}

template<typename Op, typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Op, LhsImplType, RhsImplType>>
binary_operation_function(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    return Exp<BinaryExpImpl<Op, LhsImplType, RhsImplType>>(
        Alloc::unique_construct<BinaryExpImpl<Op, LhsImplType, RhsImplType>>(
            lhs.impl_ptr(), rhs.impl_ptr()
        )
    );
}
}  // namespace

template<typename OIType>
Exp<UnaryExpImpl<Minus, OIType>>
minus(const Exp<OIType>& operand) {
    return unary_operation_function<Minus, OIType>(operand);
}
template<typename OIType>
Exp<UnaryExpImpl<Minus, OIType>> 
operator-(const Exp<OIType>& operand) {
    return minus<OIType>(operand);
}

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, RhsImplType>>
add(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    CHECK_EXP_SAME_SHAPE(lhs.impl(), rhs.impl());
    return binary_operation_function<Add, LhsImplType, RhsImplType>(lhs, rhs);
}
template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, RhsImplType>>
operator+(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    return add<LhsImplType, RhsImplType>(lhs, rhs);
}

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, RhsImplType>>
mul(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    CHECK_EXP_SAME_SHAPE(lhs.impl(), rhs.impl());
    return binary_operation_function<Mul, LhsImplType, RhsImplType>(lhs, rhs);
}
template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, RhsImplType>>
operator*(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    return mul<LhsImplType, RhsImplType>(lhs, rhs);
}

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, RhsImplType>>
sub(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    CHECK_EXP_SAME_SHAPE(lhs.impl(), rhs.impl());
    return binary_operation_function<Sub, LhsImplType, RhsImplType>(lhs, rhs);
}
template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, RhsImplType>>
operator-(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    return sub<LhsImplType, RhsImplType>(lhs, rhs);
}



}  // namespace op
}  // namespace st

#endif