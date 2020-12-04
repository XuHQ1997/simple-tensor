#ifndef EXP_OPERATOR_BASIC_OP_H
#define EXP_OPERATOR_BASIC_OP_H

#include <type_traits>

namespace st {
namespace op {

struct Minus {
    using is_elementwise = std::true_type;
    template<typename IndexType, typename OperandType>
    static data_t map(IndexType& inds, const OperandType& operand) {
        return -operand.eval(inds);
    }
};

struct Add {
    using is_elementwise = std::true_type;
    template<typename IndexType, typename LhsType, typename RhsType>
    static data_t map(IndexType& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) + rhs.eval(inds);
    }
};

struct Mul {
    using is_elementwise = std::true_type;
    template<typename IndexType, typename LhsType, typename RhsType>
    static data_t map(IndexType& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) * rhs.eval(inds);
    }
};

struct Sub {
    using is_elementwise = std::true_type;
    template<typename IndexType, typename LhsType, typename RhsType>
    static data_t map(IndexType& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) - rhs.eval(inds);
    }
};

}  // namespace op
}  // namespace st
#endif