#ifndef TENSOR_FUNCTION_H
#define TENSOR_FUNCTION_H

#include "exp/exp_impl.h"
#include "exp/grad_impl.h"
#include "exp/function.h"
#include "tensor/tensor.h"
#include "tensor/tensor_impl.h"

// It's easy to support broadcasting operations for tensors or Exps.
// But it's difficult to implement the backward of broadcasting operations.
// To make this easier, I decide to only support broadcasting for tensors,
// in other words, prohibit broadcasting when one of operands is Exp.
// So some functions and classes are specialized here for Tensors.

// template specialization for GradImpl.
namespace st {

}  // namespace st

// template specialization for ExpImpl.
namespace st {
// template<typename Op>
// class BinaryExpImpl<Op, TensorImpl, TensorImpl>
//         : public ExpImpl<BinaryExpImpl<Op, TensorImpl, TensorImpl>> {
// public:
//     BinaryExpImpl(const OperandImplPtr<TensorImpl>& lhs_ptr,
//                   const OperandImplPtr<TensorImpl>& rhs_ptr)
//             : lhs_ptr_(lhs_ptr, true),
//               rhs_ptr_(rhs_ptr, true) {}

//     index_t ndim(void) const { return Op::ndim(*lhs_ptr_, *rhs_ptr_); }
//     index_t size(index_t idx) const { return Op::size(idx, *lhs_ptr_, *rhs_ptr_); }

//     data_t eval(IndexArray& inds) const {
//         return Op::map(inds, *lhs_ptr_, *rhs_ptr_);
//     }

//     IndexArray size(void) const {
//         IndexArray shape(ndim());
//         for(index_t i = 0; i < shape.size(); ++i)
//             shape[i] = size(i);
//         return shape;
//     }

//     bool requires_grad(void) const { 
//         return lhs_ptr_->requires_grad() || rhs_ptr_->requires_grad(); 
//     }

//     template<typename GIType>
//     void backward(const GIType& grad) {
//         CHECK_EQUAL(this->gradcount(), 0, "Reused ExpImpl can't be backward.");

//         BinaryGradImpl<typename Op::LhsGrad, GIType, TensorImpl, TensorImpl> 
//         lhs_grad(grad, *lhs_ptr_, *rhs_ptr_);
//         lhs_ptr_.invoke_backward(lhs_grad);

//         BinaryGradImpl<typename Op::RhsGrad, GIType, TensorImpl, TensorImpl> 
//         rhs_grad(grad, *lhs_ptr_, *rhs_ptr_);
//         rhs_ptr_.invoke_backward(rhs_grad);
//     }
// private:
//     OperandImplPtr<TensorImpl> lhs_ptr_;
//     OperandImplPtr<TensorImpl> rhs_ptr_;
// };
}  // namespace st

// template specialization for op::functions.
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

// template specialization for Tensor::backward()
namespace st {

}  // namespace st
#endif