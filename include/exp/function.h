#ifndef EXP_FUNCTION_H
#define EXP_FUNCTION_H

#include "utils/allocator.h"
#include "utils/exception.h"
#include "exp/exp_impl.h"

#include "exp/operator/basic_op.h"
#include "exp/operator/matrix_op.h"
#include "exp/operator/log_softmax.h"

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

// function for basic operation
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
    CHECK_EXP_BROADCAST(lhs.impl(), rhs.impl());
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
    CHECK_EXP_BROADCAST(lhs.impl(), rhs.impl());
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
    CHECK_EXP_BROADCAST(lhs.impl(), rhs.impl());
    return binary_operation_function<Sub, LhsImplType, RhsImplType>(lhs, rhs);
}
template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, RhsImplType>>
operator-(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    return sub<LhsImplType, RhsImplType>(lhs, rhs);
}

template<typename OIType>
Exp<UnaryExpImpl<ReLU, OIType>>
relu(const Exp<OIType>& operand) {
    return unary_operation_function<ReLU, OIType>(operand);
}

template<typename OIType>
Exp<UnaryExpImpl<Sigmoid, OIType>>
sigmoid(const Exp<OIType>& operand) {
    return unary_operation_function<Sigmoid, OIType>(operand);
}

// function for matrix operation
template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<MatrixMul, LhsImplType, RhsImplType>>
matrix_mul(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    auto lhs_impl = lhs.impl();
    auto rhs_impl = rhs.impl();
    CHECK_TRUE(lhs_impl.ndim() == 2 && rhs_impl.ndim() == 2, 
        "Matrices expected, got %dD and %dD Tensor。", 
        lhs_impl.ndim(), rhs_impl.ndim());
    CHECK_EQUAL(lhs_impl.size(1), rhs_impl.size(0), 
        "Size mismatch, m1: [%d, %d], m2: [%d, %d].",
        lhs_impl.size(0), lhs_impl.size(1), rhs_impl.size(0), rhs_impl.size(1));
    return binary_operation_function<MatrixMul, LhsImplType, RhsImplType>(lhs, rhs);
}

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<BatchMatrixMul, LhsImplType, RhsImplType>>
batch_matrix_mul(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    auto lhs_impl = lhs.impl();
    auto rhs_impl = rhs.impl();
    CHECK_TRUE(lhs_impl.ndim() == 3 && rhs_impl.ndim() == 3, 
        "Baths of Matrices expected, got %dD and %dD Tensor。", 
        lhs_impl.ndim(), rhs_impl.ndim());
    CHECK_TRUE(lhs_impl.size(0) == rhs_impl.size(0) 
            || lhs_impl.size(0) == 1 
            || rhs_impl.size(0) == 1,
        "Bath sizes, %d and %d, doesn't match.",
        lhs_impl.size(0), rhs_impl.size(0));
    CHECK_EQUAL(lhs_impl.size(2), rhs_impl.size(1),
        "Size mismatch, m1: [%d, %d], m2: [%d, %d].",
        lhs_impl.size(1), lhs_impl.size(2), rhs_impl.size(1), rhs_impl.size(2));
    return binary_operation_function<BatchMatrixMul, LhsImplType, RhsImplType>(lhs, rhs);
}

// function for log_softmax
template<typename OIType>
Exp<UnaryExpImpl<LogSoftmax, OIType>>
log_softmax(const Exp<OIType>& operand) {
    CHECK_EQUAL(operand.impl().ndim(), 2, 
        "log_softmax Only supported for 2D Tensor, but got a %dD one", 
        operand.impl().ndim());
    return unary_operation_function<LogSoftmax, OIType>(operand);
}


// function for nll_loss
template<typename OIType>
Exp<UnaryExpImpl<MeanReduce, OIType>>
mean(const Exp<OIType>& operand, index_t dim) {
    CHECK_IN_RANGE(dim, 0, operand.impl().ndim(), 
        "Dimension out of range (expected to be in range of [0, %d), but got %d)",
        operand.impl().ndim(), dim);
    return Exp<UnaryExpImpl<MeanReduce, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<MeanReduce, OIType>>(
            operand.impl_ptr(), dim
        )
    );
}

template<typename OIType>
Exp<UnaryExpImpl<NLLLoss, OIType>>
nll_loss(const Exp<OIType>& operand, 
         const std::shared_ptr<index_t>& labels_ptr, 
         index_t n_label=-1) {
    CHECK_EQUAL(operand.impl().ndim(), 2, 
        "NLL Loss is only supported for 2D Tensor, but got %dD one.", 
        operand.impl().ndim());

    index_t n_batch = operand.impl().size(0);
    index_t n_cls = operand.impl().size(1);
    CHECK_TRUE(n_label == -1 || n_label == n_batch,
        "Batch size mismatch, x: %d, labels: %d", n_batch, n_label);

    auto labels = labels_ptr.get();
    for(index_t i = 0; i < n_batch; ++i)
        CHECK_IN_RANGE(labels[i], 0, n_cls,
            "%d classes got label of %d", n_cls, labels[i]);

    return Exp<UnaryExpImpl<NLLLoss, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<NLLLoss, OIType>>(
            operand.impl_ptr(), labels_ptr
        )
    );
}

}  // namespace op
}  // namespace st

#endif