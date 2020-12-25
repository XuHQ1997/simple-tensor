#ifndef EXP_FUNCTION_H
#define EXP_FUNCTION_H

#include <type_traits>
#include <memory>
#include <cstring>

#include "utils/allocator.h"
#include "utils/exception.h"
#include "exp/exp_impl.h"
#include "exp/exp.h"

#include "exp/operator/basic_op.h"
#include "exp/operator/matrix_op.h"
#include "exp/operator/reduce_op.h"
#include "exp/operator/nll_loss.h"
#include "exp/operator/log_softmax.h"
#include "exp/operator/conv.h"

namespace st {

template<typename ImplType> class Exp;

namespace op {

template<typename Op, typename OIType>  // OIType = OperandImplType
Exp<UnaryExpImpl<Op, OIType>> 
__unary_operation_function(const Exp<OIType>& operand) {
    return Exp<UnaryExpImpl<Op, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<Op, OIType>>(
            operand.impl_ptr()
        )
    );
}

template<typename Op, typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Op, LhsImplType, RhsImplType>>
__binary_operation_function(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    return Exp<BinaryExpImpl<Op, LhsImplType, RhsImplType>>(
        Alloc::unique_construct<BinaryExpImpl<Op, LhsImplType, RhsImplType>>(
            lhs.impl_ptr(), rhs.impl_ptr()
        )
    );
}

template<typename LhsImplType, typename RhsImplType>
typename std::enable_if<LhsImplType::op::Grad::allow_broadcast::value 
                     && RhsImplType::op::Grad::allow_broadcast::value,
                        void>::type
__check_broadcast(const LhsImplType& lhs, const RhsImplType& rhs) {
    CHECK_EXP_BROADCAST(lhs, rhs);
}

template<typename LhsImplType, typename RhsImplType>
typename std::enable_if<!(LhsImplType::op::Grad::allow_broadcast::value 
                       && RhsImplType::op::Grad::allow_broadcast::value),
                        void>::type
__check_broadcast(const LhsImplType& lhs, const RhsImplType& rhs) {
    CHECK_EXP_SAME_SHAPE(lhs, rhs);
}

// function for basic operation
template<typename OIType>
Exp<UnaryExpImpl<Minus, OIType>>
minus(const Exp<OIType>& operand) {
    return __unary_operation_function<Minus, OIType>(operand);
}
template<typename OIType>
Exp<UnaryExpImpl<Minus, OIType>> 
operator-(const Exp<OIType>& operand) {
    return minus<OIType>(operand);
}

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, RhsImplType>>
add(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    __check_broadcast(lhs.impl(), rhs.impl());
    return __binary_operation_function<Add, LhsImplType, RhsImplType>(lhs, rhs);
}
template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Add, LhsImplType, RhsImplType>>
operator+(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    return add<LhsImplType, RhsImplType>(lhs, rhs);
}

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, RhsImplType>>
mul(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    __check_broadcast(lhs.impl(), rhs.impl());
    return __binary_operation_function<Mul, LhsImplType, RhsImplType>(lhs, rhs);
}
template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Mul, LhsImplType, RhsImplType>>
operator*(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    return mul<LhsImplType, RhsImplType>(lhs, rhs);
}

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, RhsImplType>>
sub(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    __check_broadcast(lhs.impl(), rhs.impl());
    return __binary_operation_function<Sub, LhsImplType, RhsImplType>(lhs, rhs);
}
template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<Sub, LhsImplType, RhsImplType>>
operator-(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    return sub<LhsImplType, RhsImplType>(lhs, rhs);
}

template<typename OIType>
Exp<UnaryExpImpl<ReLU, OIType>>
relu(const Exp<OIType>& operand) {
    return __unary_operation_function<ReLU, OIType>(operand);
}

template<typename OIType>
Exp<UnaryExpImpl<Sigmoid, OIType>>
sigmoid(const Exp<OIType>& operand) {
    return __unary_operation_function<Sigmoid, OIType>(operand);
}

// function for matrix operation
template<typename OIType>
Exp<UnaryExpImpl<MatrixTranspose, OIType>>
matrix_transpose(const Exp<OIType>& operand) {
    CHECK_EQUAL(operand.impl().ndim(), 2,
        "Matrix Transpose is only supported for 2D Tensor, but got %dD one",
        operand.impl().ndim());
    return __unary_operation_function<MatrixTranspose, OIType>(operand);
}

template<typename OIType>
Exp<UnaryExpImpl<BatchMatrixTranspose, OIType>>
batch_matrix_transpose(const Exp<OIType>& operand) {
    CHECK_EQUAL(operand.impl().ndim(), 3,
        "Batch Matrix Transpose is only supported for 3D Tensor, but got %dD one",
        operand.impl().ndim());
    return __unary_operation_function<BatchMatrixTranspose, OIType>(operand);
}

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<MatrixMul, LhsImplType, RhsImplType>>
matrix_mul(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    auto& lhs_impl = lhs.impl();
    auto& rhs_impl = rhs.impl();
    CHECK_TRUE(lhs_impl.ndim() == 2 && rhs_impl.ndim() == 2, 
        "Matrices expected, got %dD and %dD Tensor。", 
        lhs_impl.ndim(), rhs_impl.ndim());
    CHECK_EQUAL(lhs_impl.size(1), rhs_impl.size(0), 
        "Size mismatch, m1: [%d, %d], m2: [%d, %d].",
        lhs_impl.size(0), lhs_impl.size(1), rhs_impl.size(0), rhs_impl.size(1));
    return __binary_operation_function<MatrixMul, LhsImplType, RhsImplType>(lhs, rhs);
}

template<typename LhsImplType, typename RhsImplType>
Exp<BinaryExpImpl<BatchMatrixMul, LhsImplType, RhsImplType>>
batch_matrix_mul(const Exp<LhsImplType>& lhs, const Exp<RhsImplType>& rhs) {
    auto& lhs_impl = lhs.impl();
    auto& rhs_impl = rhs.impl();
    CHECK_TRUE(lhs_impl.ndim() == 3 && rhs_impl.ndim() == 3, 
        "Baths of Matrices expected, got %dD and %dD Tensor。", 
        lhs_impl.ndim(), rhs_impl.ndim());
    CHECK_TRUE(lhs_impl.size(0) == rhs_impl.size(0),
        "Bath sizes, %d and %d, doesn't match.",
        lhs_impl.size(0), rhs_impl.size(0));
    CHECK_EQUAL(lhs_impl.size(2), rhs_impl.size(1),
        "Size mismatch, m1: [%d, %d], m2: [%d, %d].",
        lhs_impl.size(1), lhs_impl.size(2), rhs_impl.size(1), rhs_impl.size(2));
    return __binary_operation_function<BatchMatrixMul, LhsImplType, RhsImplType>(lhs, rhs);
}

// function for log_softmax
template<typename OIType>
Exp<UnaryExpImpl<LogSoftmax, OIType>>
log_softmax(const Exp<OIType>& operand) {
    CHECK_EQUAL(operand.impl().ndim(), 2, 
        "log_softmax Only supported for 2D Tensor, but got a %dD one", 
        operand.impl().ndim());
    return __unary_operation_function<LogSoftmax, OIType>(operand);
}


// function for reduce operator
template<typename OIType>
Exp<UnaryExpImpl<Mean, OIType>>
mean(const Exp<OIType>& operand, index_t dim) {
    CHECK_IN_RANGE(dim, 0, operand.impl().ndim(), 
        "Dimension out of range (expected to be in range of [0, %d), but got %d)",
        operand.impl().ndim(), dim);
    return Exp<UnaryExpImpl<Mean, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<Mean, OIType>>(
            operand.impl_ptr(), dim
        )
    );
}

template<typename OIType>
Exp<UnaryExpImpl<Max, OIType>>
max(const Exp<OIType>& operand, index_t dim) {
    CHECK_IN_RANGE(dim, 0, operand.impl().ndim(), 
        "Dimension out of range (expected to be in range of [0, %d), but got %d)",
        operand.impl().ndim(), dim);
    return Exp<UnaryExpImpl<Max, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<Max, OIType>>(
            operand.impl_ptr(), dim
        )
    );
}

template<typename OIType>
Exp<UnaryExpImpl<Argmax, OIType>>
argmax(const Exp<OIType>& operand, index_t dim) {
    CHECK_IN_RANGE(dim, 0, operand.impl().ndim(), 
        "Dimension out of range (expected to be in range of [0, %d), but got %d)",
        operand.impl().ndim(), dim);
    return Exp<UnaryExpImpl<Argmax, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<Argmax, OIType>>(
            operand.impl_ptr(), dim
        )
    );
}

// function for nll_loss
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

template<typename OIType>
Exp<UnaryExpImpl<NLLLoss, OIType>>
nll_loss(const Exp<OIType>& operand, 
         const index_t* labels, 
         index_t n_label=-1) {
    CHECK_EQUAL(operand.impl().ndim(), 2, 
        "NLL Loss is only supported for 2D Tensor, but got %dD one.", 
        operand.impl().ndim());

    index_t n_batch = operand.impl().size(0);
    index_t n_cls = operand.impl().size(1);
    CHECK_TRUE(n_label == -1 || n_label == n_batch,
        "Batch size mismatch, x: %d, labels: %d", n_batch, n_label);

    for(index_t i = 0; i < n_batch; ++i)
        CHECK_IN_RANGE(labels[i], 0, n_cls,
            "%d classes got label of %d", n_cls, labels[i]);

    std::shared_ptr<index_t> labels_ptr = 
        Alloc::shared_allocate<index_t>(n_batch * sizeof(index_t));
    std::memcpy(labels_ptr.get(), labels, n_batch * sizeof(index_t));

    return Exp<UnaryExpImpl<NLLLoss, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<NLLLoss, OIType>>(
            operand.impl_ptr(), labels_ptr
        )
    );
}

// function for conv
template<typename OIType>
Exp<UnaryExpImpl<Img2col, OIType>>
img2col(const Exp<OIType>& operand, const Img2col::Wsize& kernel_size,
        const Img2col::Wsize& stride_size, const Img2col::Wsize& padding_size) {
    CHECK_EQUAL(operand.impl().ndim(), 4, 
        "Img2col is only supported for 4D Tensor, but got a %dD one", 
        operand.impl().ndim());
    CHECK_INDEX_VALID(kernel_size.first, "Invalid kernel_size.");
    CHECK_INDEX_VALID(kernel_size.second, "Invalid kernel_size.");
    CHECK_IN_RANGE(stride_size.first, 1, INDEX_MAX, "Invalid stride_size.");
    CHECK_IN_RANGE(stride_size.second, 1, INDEX_MAX, "Invalid stride_size.");
    CHECK_INDEX_VALID(padding_size.first, "Invalid padding_size.");
    CHECK_INDEX_VALID(padding_size.second, "Invalid padding_size.");
    CHECK_INDEX_VALID(operand.impl().size(2) + 2*padding_size.first - kernel_size.first, 
        "Kernel size (%d %d) is too large", kernel_size.first, kernel_size.second);
    CHECK_INDEX_VALID(operand.impl().size(3) + 2*padding_size.second - kernel_size.second, 
        "Kernel size (%d %d) is too large", kernel_size.first, kernel_size.second);
    return Exp<UnaryExpImpl<Img2col, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<Img2col, OIType>>(
            operand.impl_ptr(), kernel_size, stride_size, padding_size 
        )
    );
}

template<typename OIType>
Exp<UnaryExpImpl<MaxPool2d, OIType>>
max_pool2d(const Exp<OIType>& operand, const MaxPool2d::Wsize& kernel_size,
           const MaxPool2d::Wsize& stride_size, const MaxPool2d::Wsize& padding_size) {
    CHECK_EQUAL(operand.impl().ndim(), 4, 
        "MaxPool2d is only supported for 4D Tensor, but got a %dD one", 
        operand.impl().ndim());
    CHECK_INDEX_VALID(kernel_size.first, "Invalid kernel_size.");
    CHECK_INDEX_VALID(kernel_size.second, "Invalid kernel_size.");
    CHECK_IN_RANGE(stride_size.first, 1, INDEX_MAX, "Invalid stride_size.");
    CHECK_IN_RANGE(stride_size.second, 1, INDEX_MAX, "Invalid stride_size.");
    CHECK_INDEX_VALID(padding_size.first, "Invalid padding_size.");
    CHECK_INDEX_VALID(padding_size.second, "Invalid padding_size.");
    CHECK_INDEX_VALID(operand.impl().size(2) + 2*padding_size.first - kernel_size.first, 
        "Kernel size (%d %d) is too large", kernel_size.first, kernel_size.second);
    CHECK_INDEX_VALID(operand.impl().size(3) + 2*padding_size.second - kernel_size.second, 
        "Kernel size (%d %d) is too large", kernel_size.first, kernel_size.second);
    return Exp<UnaryExpImpl<MaxPool2d, OIType>>(
        Alloc::unique_construct<UnaryExpImpl<MaxPool2d, OIType>>(
            operand.impl_ptr(), kernel_size, stride_size, padding_size 
        )
    );
}

inline Exp<UnaryExpImpl<Constant, data_t>>
constant(data_t value, IndexArray&& size) {
    return Exp<UnaryExpImpl<Constant, data_t>>(
        Alloc::unique_construct<UnaryExpImpl<Constant, data_t>>(
            value, std::move(size)
        )
    );
}

}  // namespace op
}  // namespace st

#endif