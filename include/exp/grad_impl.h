#ifndef EXP_GRAD_IMPL_H
#define EXP_GRAD_IMPL_H

#include "utils/base_config.h"
#include "utils/array.h"

#include "exp/operator/log_softmax.h"
#include "exp/operator/constant.h"
#include "exp/operator/reduce_op.h"
#include "exp/operator/nll_loss.h"
#include "exp/operator/conv.h"

namespace st {

// GradImpl is the template expression used in backward
template<typename ImplType>
class GradImpl {
public:
    const ImplType& self(void) const {
        return *static_cast<ImplType*>(this);
    }
};

// OIType = OperandImplType; GIType = GradImplType
template<typename Op, typename GIType, typename OIType>
class UnaryGradImpl : public GradImpl<UnaryGradImpl<Op, GIType, OIType>>{
public:
    UnaryGradImpl(const GIType& grad, const OIType& operand)
            : grad_(grad), operand_(operand) {}

    data_t eval(IndexArray& inds) const {
        return Op::map(inds, grad_, operand_);
    }
private:
    const GIType& grad_;
    const OIType& operand_;
};

template<typename Op, typename GIType, typename LhsImplType, typename RhsImplType>
class BinaryGradImpl 
        : public GradImpl<BinaryGradImpl<Op, GIType, LhsImplType, RhsImplType>> {
public:
    BinaryGradImpl(const GIType& grad, const LhsImplType& lhs, 
                   const RhsImplType& rhs)
            : grad_(grad), lhs_(lhs), rhs_(rhs) {}

    data_t eval(IndexArray& inds) const {
        return Op::map(inds, grad_, lhs_, rhs_);
    }
private:
    const GIType& grad_;
    const LhsImplType& lhs_;
    const RhsImplType& rhs_;
};
}  // namespace st


// Template specialization for UnaryGradImpl and BinaryGradImpl
namespace st {

template<typename GIType, typename OIType>
class UnaryGradImpl<typename op::LogSoftmax::Grad, GIType, OIType>
        : public GradImpl<UnaryGradImpl<typename op::LogSoftmax::Grad, GIType, OIType>> {
public:
    UnaryGradImpl(const GIType& grad, const OIType& operand, 
                 data_t* batch_sum_exp, data_t* batch_max_cls)
            : grad_(grad), operand_(operand),
              batch_sum_exp_(batch_sum_exp),
              batch_max_cls_(batch_max_cls) {}

    data_t eval(IndexArray& inds) const {
        return op::LogSoftmax::Grad::map(
            inds, grad_, operand_, batch_sum_exp_, batch_max_cls_
        );
    }
private:
    const GIType& grad_;
    const OIType& operand_;

    data_t* batch_sum_exp_;
    data_t* batch_max_cls_;
};

template<typename GIType, typename OIType>
class UnaryGradImpl<typename op::Mean::Grad, GIType, OIType>
        : public GradImpl<UnaryGradImpl<typename op::Mean::Grad, GIType, OIType>> {
public:
    UnaryGradImpl(const GIType& grad, const OIType& operand,
                  index_t reduce_dim)
            : grad_(grad), operand_(operand),
              reduce_dim_(reduce_dim) {}
    
    data_t eval(IndexArray& inds) const {
        return op::Mean::Grad::map(
            inds, grad_, operand_, reduce_dim_
        );
    }
private:
    const GIType& grad_;
    const OIType& operand_;
    
    index_t reduce_dim_;
};

template<typename GIType, typename OIType>
class UnaryGradImpl<typename op::Max::Grad, GIType, OIType>
        : public GradImpl<UnaryGradImpl<typename op::Max::Grad, GIType, OIType>> {
public:
    UnaryGradImpl(const GIType& grad, const OIType& operand,
                 index_t reduce_dim)
            : grad_(grad), operand_(operand),
              reduce_dim_(reduce_dim) {}
    
    data_t eval(IndexArray& inds) const {
        return op::Max::Grad::map(
            inds, grad_, operand_, reduce_dim_
        );
    }
private:
    const GIType& grad_;
    const OIType& operand_;

    index_t reduce_dim_;
};

template<typename GIType, typename OIType>
class UnaryGradImpl<typename op::NLLLoss::Grad, GIType, OIType>
        : public GradImpl<UnaryGradImpl<typename op::NLLLoss::Grad, GIType, OIType>> {
public:
    UnaryGradImpl(const GIType& grad, const OIType& operand,
                  const index_t* batch_label)
            : grad_(grad), operand_(operand),
              batch_label_(batch_label) {}
    
    data_t eval(IndexArray& inds) const {
        return op::NLLLoss::Grad::map(
            inds, grad_, operand_, batch_label_
        );
    }
private:
    const GIType& grad_;
    const OIType& operand_;

    const index_t* batch_label_;
};

template<typename GIType, typename OIType>
class UnaryGradImpl<typename op::Img2col::Grad, GIType, OIType>
        : public GradImpl<UnaryGradImpl<typename op::Img2col::Grad, GIType, OIType>> {
public:
    using Wsize = typename op::Img2col::Wsize;

    UnaryGradImpl(const GIType& grad, const OIType& operand,
                  const Wsize& kernel_size, const Wsize& stride_size,
                  const Wsize& padding_size, const Wsize& out_size)
            : grad_(grad), operand_(operand),
              kernel_size_(kernel_size), stride_size_(stride_size),
              padding_size_(padding_size), out_size_(out_size) {}
    
    data_t eval(IndexArray& inds) const {
        return op::Img2col::Grad::map(
            inds, grad_, operand_, kernel_size_, stride_size_,
            padding_size_, out_size_
        );
    }

private:
    const GIType& grad_;
    const OIType& operand_;

    Wsize kernel_size_;
    Wsize stride_size_;
    Wsize padding_size_;
    Wsize out_size_;
};

template<>
class UnaryGradImpl<op::Constant, void, data_t>
        : public GradImpl<UnaryGradImpl<op::Constant, void, data_t>> {
public:    
    UnaryGradImpl(data_t value)
            : value_(value) {}

    data_t eval(IndexArray& inds) const {
        return op::Constant::map(inds, value_);
    }
private:
    data_t value_;
};

}  // namespace st
#endif