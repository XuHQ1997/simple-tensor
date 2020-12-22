#ifndef TENSOR_BROADCASTING_H
#define TENSOR_BROADCASTING_H

#include "exp/exp_impl.h"
#include "exp/grad_impl.h"
#include "exp/function.h"
#include "tensor/tensor.h"
#include "tensor/tensor_impl.h"

// It's easy to support broadcasting operations for tensors or Exps.
// But it's difficult to implement the backward of broadcasting operations.
// To make this easier, I decide to only support broadcasting for tensors,
// in other words, prohibit broadcasting when anyone of operand is Exp.
// So some functions and classes are specialized here for Tensors.

// template specialization for GradImpl.
namespace st {
template<typename GIType>
class BinaryGradImpl<typename op::Add::LhsGrad, GIType, TensorImpl, TensorImpl>
        : public GradImpl<
            BinaryGradImpl<
                typename op::Add::LhsGrad, GIType, TensorImpl, TensorImpl>> {
public:
    BinaryGradImpl(const GIType& grad, const TensorImpl& lhs, 
                   const TensorImpl& rhs)
            : grad_(grad), lhs_(lhs), rhs_(rhs) {
        CHECK_EQUAL(lhs_.ndim(), rhs_.ndim(), 
            "Backward of broadcasting is supported only when the dimensions \
            of operands are equal, but got %dD and %dD.",
            lhs_.ndim(), rhs_.ndim());
    }

    IndexArray grad_size(void) const { 
        return grad_.grad_size();
    }

    data_t eval(IndexArray& inds) const {
        return op::Add::LhsGrad::map(inds, grad_, lhs_, rhs_);
    }
private:
    const GIType& grad_;
    const TensorImpl& lhs_;
    const TensorImpl& rhs_;
};

template<typename GIType>
class BinaryGradImpl<typename op::Add::RhsGrad, GIType, TensorImpl, TensorImpl>
        : public GradImpl<
            BinaryGradImpl<
                typename op::Add::RhsGrad, GIType, TensorImpl, TensorImpl>> {
public:
    BinaryGradImpl(const GIType& grad, const TensorImpl& lhs, 
                   const TensorImpl& rhs)
            : grad_(grad), lhs_(lhs), rhs_(rhs) {
        CHECK_EQUAL(lhs_.ndim(), rhs_.ndim(), 
            "Backward of broadcasting is supported only when the dimensions \
            of operands are equal, but got %dD and %dD.",
            lhs_.ndim(), rhs_.ndim());
    }

    IndexArray grad_size(void) const { 
        return grad_.grad_size();
    }

    data_t eval(IndexArray& inds) const {
        return op::Add::RhsGrad::map(inds, grad_, lhs_, rhs_);
    }
private:
    const GIType& grad_;
    const TensorImpl& lhs_;
    const TensorImpl& rhs_;
};

template<typename GIType>
class BinaryGradImpl<typename op::Mul::LhsGrad, GIType, TensorImpl, TensorImpl>
        : public GradImpl<
            BinaryGradImpl<
                typename op::Mul::LhsGrad, GIType, TensorImpl, TensorImpl>> {
public:
    BinaryGradImpl(const GIType& grad, const TensorImpl& lhs, 
                   const TensorImpl& rhs)
            : grad_(grad), lhs_(lhs), rhs_(rhs) {
        CHECK_EQUAL(lhs_.ndim(), rhs_.ndim(), 
            "Backward of broadcasting is supported only when the dimensions \
            of operands are equal, but got %dD and %dD.",
            lhs_.ndim(), rhs_.ndim());
    }

    IndexArray grad_size(void) const { 
        return grad_.grad_size();
    }

    data_t eval(IndexArray& inds) const {
        return op::Mul::LhsGrad::map(inds, grad_, lhs_, rhs_);
    }
private:
    const GIType& grad_;
    const TensorImpl& lhs_;
    const TensorImpl& rhs_;
};

template<typename GIType>
class BinaryGradImpl<typename op::Mul::RhsGrad, GIType, TensorImpl, TensorImpl>
        : public GradImpl<
            BinaryGradImpl<
                typename op::Mul::RhsGrad, GIType, TensorImpl, TensorImpl>> {
public:
    BinaryGradImpl(const GIType& grad, const TensorImpl& lhs, 
                   const TensorImpl& rhs)
            : grad_(grad), lhs_(lhs), rhs_(rhs) {
        CHECK_EQUAL(lhs_.ndim(), rhs_.ndim(), 
            "Backward of broadcasting is supported only when the dimensions \
            of operands are equal, but got %dD and %dD.",
            lhs_.ndim(), rhs_.ndim());
    }

    IndexArray grad_size(void) const { 
        return grad_.grad_size();
    }

    data_t eval(IndexArray& inds) const {
        return op::Mul::RhsGrad::map(inds, grad_, lhs_, rhs_);
    }
private:
    const GIType& grad_;
    const TensorImpl& lhs_;
    const TensorImpl& rhs_;
};

template<typename GIType>
class BinaryGradImpl<typename op::Sub::LhsGrad, GIType, TensorImpl, TensorImpl>
        : public GradImpl<
            BinaryGradImpl<
                typename op::Sub::LhsGrad, GIType, TensorImpl, TensorImpl>> {
public:
    BinaryGradImpl(const GIType& grad, const TensorImpl& lhs, 
                   const TensorImpl& rhs)
            : grad_(grad), lhs_(lhs), rhs_(rhs) {
        CHECK_EQUAL(lhs_.ndim(), rhs_.ndim(), 
            "Backward of broadcasting is supported only when the dimensions \
            of operands are equal, but got %dD and %dD.",
            lhs_.ndim(), rhs_.ndim());
    }

    IndexArray grad_size(void) const { 
        return grad_.grad_size();
    }

    data_t eval(IndexArray& inds) const {
        return op::Sub::LhsGrad::map(inds, grad_, lhs_, rhs_);
    }
private:
    const GIType& grad_;
    const TensorImpl& lhs_;
    const TensorImpl& rhs_;
};

template<typename GIType>
class BinaryGradImpl<typename op::Sub::RhsGrad, GIType, TensorImpl, TensorImpl>
        : public GradImpl<
            BinaryGradImpl<
                typename op::Sub::RhsGrad, GIType, TensorImpl, TensorImpl>> {
public:
    BinaryGradImpl(const GIType& grad, const TensorImpl& lhs, 
                   const TensorImpl& rhs)
            : grad_(grad), lhs_(lhs), rhs_(rhs) {
        CHECK_EQUAL(lhs_.ndim(), rhs_.ndim(), 
            "Backward of broadcasting is supported only when the dimensions \
            of operands are equal, but got %dD and %dD.",
            lhs_.ndim(), rhs_.ndim());
    }

    IndexArray grad_size(void) const { 
        return grad_.grad_size();
    }

    data_t eval(IndexArray& inds) const {
        return op::Sub::RhsGrad::map(inds, grad_, lhs_, rhs_);
    }
private:
    const GIType& grad_;
    const TensorImpl& lhs_;
    const TensorImpl& rhs_;
};
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
}  // namespace op
}  // namespace st
#endif