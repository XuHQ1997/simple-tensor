#ifndef TENSOR_FUNCTION_H
#define TENSOR_FUNCTION_H

#include "exp/function.h"
#include "tensor/tensor.h"
#include "tensor/tensor_impl.h"

// It's easy to support broadcasting operations for tensors or Exps.
// But it's difficult to implement the backward of broadcasting operations.
// To make this easier, I decide to only support broadcasting for tensors,
// in other words, prohibit broadcasting when one of operands is Exp.
// So some function is specialized here for Tensors.
namespace st {
namespace op {

template<>
inline Exp<BinaryExpImpl<Add, TensorImpl, TensorImpl>>
add(const Exp<TensorImpl>& lhs, const Exp<TensorImpl>& rhs) {
    CHECK_EXP_BROADCAST(lhs.impl(), rhs.impl());
    return binary_operation_function<Add, TensorImpl, TensorImpl>(lhs, rhs);
}

template<>
inline Exp<BinaryExpImpl<Mul, TensorImpl, TensorImpl>>
mul(const Exp<TensorImpl>& lhs, const Exp<TensorImpl>& rhs) {
    CHECK_EXP_BROADCAST(lhs.impl(), rhs.impl());
    return binary_operation_function<Mul, TensorImpl, TensorImpl>(lhs, rhs);
}

template<>
inline Exp<BinaryExpImpl<Sub, TensorImpl, TensorImpl>>
sub(const Exp<TensorImpl>& lhs, const Exp<TensorImpl>& rhs) {
    CHECK_EXP_BROADCAST(lhs.impl(), rhs.impl());
    return binary_operation_function<Sub, TensorImpl, TensorImpl>(lhs, rhs);
}

template<>
inline Exp<BinaryExpImpl<BatchMatrixMul, TensorImpl, TensorImpl>>
batch_matrix_mul(const Exp<TensorImpl>& lhs, const Exp<TensorImpl>& rhs) {
    auto& lhs_impl = lhs.impl();
    auto& rhs_impl = rhs.impl();
    CHECK_TRUE(lhs_impl.ndim() == 3 && rhs_impl.ndim() == 3, 
        "Baths of Matrices expected, got %dD and %dD Tensor.", 
        lhs_impl.ndim(), rhs_impl.ndim());
    CHECK_TRUE(lhs_impl.size(0) == rhs_impl.size(0) 
            || lhs_impl.size(0) == 1 
            || rhs_impl.size(0) == 1,
        "Bath sizes, %d and %d, doesn't match.",
        lhs_impl.size(0), rhs_impl.size(0));
    CHECK_EQUAL(lhs_impl.size(2), rhs_impl.size(1),
        "Size mismatch, m1: [%d, %d], m2: [%d, %d].",
        lhs_impl.size(1), lhs_impl.size(2), rhs_impl.size(1), rhs_impl.size(2));
    return binary_operation_function<BatchMatrixMul, TensorImpl, TensorImpl>(lhs, rhs);
}

}  // namespace op
}  // namespace st

#endif