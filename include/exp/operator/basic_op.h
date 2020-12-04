#ifndef EXP_OPERATOR_BASIC_OP_H
#define EXP_OPERATOR_BASIC_OP_H

#include <type_traits>
#include <algorithm>

namespace st {
namespace op {

struct UnaryBasicOperator {
    using is_elementwise = std::true_type;
    template<typename OperandType>
    static index_t ndim(const OperandType& operand) { 
        return operand.ndim(); 
    }
    template<typename OperandType>
    static index_t size(index_t idx, const OperandType& operand) {
        return operand.size(idx);
    }
};

struct BinaryBasicOperator {
    using is_elementwise = std::true_type;
    template<typename LhsType, typename RhsType>
    static index_t ndim(const LhsType& lhs, const RhsType& rhs) { 
        return std::max(lhs.ndim(), rhs.ndim()); 
    }
    template<typename LhsType, typename RhsType>
    static index_t size(index_t idx, const LhsType& lhs, const RhsType& rhs) {
        return std::max(lhs.size(idx), rhs.size(idx));
    }
};

struct Minus: public UnaryBasicOperator {
    template<typename IndexType, typename OperandType>
    static data_t map(IndexType& inds, const OperandType& operand) {
        return -operand.eval(inds);
    }
};

struct Add : public BinaryBasicOperator {
    template<typename IndexType, typename LhsType, typename RhsType>
    static data_t map(IndexType& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) + rhs.eval(inds);
    }
};

struct Mul : public BinaryBasicOperator {
    template<typename IndexType, typename LhsType, typename RhsType>
    static data_t map(IndexType& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) * rhs.eval(inds);
    }
};

struct Sub : public BinaryBasicOperator{
    template<typename IndexType, typename LhsType, typename RhsType>
    static data_t map(IndexType& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) - rhs.eval(inds);
    }
};

}  // namespace op
}  // namespace st
#endif