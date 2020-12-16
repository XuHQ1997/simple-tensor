#ifndef EXP_GRAD_IMPL_H
#define EXP_GRAD_IMPL_H

#include "utils/base_config.h"
#include "utils/array.h"
#include "exp/operator/constant.h"


namespace st {

// GradImpl is the template expression used in backward
template<typename ImplType>
class GradImpl {
public:
    const ImplType& self(void) const {
        return *static_cast<ImplType*>(this);
    }
};

template<typename Op, typename OIType>  // OIType = Operand Impl Type
class UnaryGradImpl : public GradImpl<UnaryGradImpl<Op, OIType>>{
public:
    UnaryGradImpl(OIType& operand)
            : operand_(operand) {}

    data_t eval(IndexArray& inds) const {
        return Op::Grad::map(inds, operand_);
    }
private:
    OIType& operand_;
};

template<typename Op, typename LhsImplType, typename RhsImplType>
class BinaryGradImpl 
        : public GradImpl<BinaryGradImpl<Op, LhsImplType, RhsImplType>> {
public:
    BinaryGradImpl(LhsImplType& lhs, RhsImplType& rhs)
            : lhs_(lhs), rhs_(rhs) {}

    data_t eval(IndexArray& inds) const {
        return Op::map(inds, lhs_, rhs_);
    }
private:
    LhsImplType& lhs_;
    RhsImplType& rhs_;
};
}  // namespace st


// Template specialization for UnaryGradImpl and BinaryGradImpl
namespace st {


template<>
class UnaryGradImpl<op::Constant, data_t>
        : public GradImpl<UnaryGradImpl<op::Constant, data_t>> {
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