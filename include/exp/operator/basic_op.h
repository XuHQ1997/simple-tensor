#ifndef EXP_OPERATOR_BASIC_OP_H
#define EXP_OPERATOR_BASIC_OP_H

#include <type_traits>

#include "utils/base_config.h"

namespace st {
namespace op {

struct UnaryBasicOperator {
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
    template<typename LhsType, typename RhsType>
    static index_t ndim(const LhsType& lhs, const RhsType& rhs) { 
        return std::max(lhs.ndim(), rhs.ndim()); 
    }
    template<typename LhsType, typename RhsType>
    static index_t size(index_t idx, const LhsType& lhs, const RhsType& rhs) {
        if(idx >= lhs.ndim()) return rhs.size(idx);
        if(idx >= rhs.ndim()) return lhs.size(idx);
        return std::max(lhs.size(idx), rhs.size(idx));
    }
};

struct Minus: public UnaryBasicOperator {
    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand) {
        return -operand.eval(inds);
    }

    struct Grad {
        using allow_broadcast = std::true_type;
        using is_lhs = std::false_type;
        using is_rhs = std::false_type;

        template<typename GradType, typename OperandType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const OperandType& oeprand) {
            return -grad.eval(inds);
        }
    };
};

struct Add : public BinaryBasicOperator {
    template<typename LhsType, typename RhsType>
    static data_t map(IndexArray& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) + rhs.eval(inds);
    }

    struct Grad {
        using allow_broadcast = std::true_type;

        struct Lhs {
            using allow_broadcast = allow_broadcast;
            using is_lhs = std::true_type;
            using is_rhs = std::false_type;

            template<typename GradType, typename LhsType, typename RhsType>
            static data_t map(IndexArray& inds, const GradType& grad,
                            const LhsType& lhs, const RhsType& rhs) {
                return grad.eval(inds);
            }

            // template<typename LhsType, typename RhsType>
            // static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            //     return lhs.size();
            // }
        };

        struct Rhs {
            using allow_broadcast = allow_broadcast;
            using is_lhs = std::false_type;
            using is_rhs = std::true_type;

            template<typename GradType, typename LhsType, typename RhsType>
            static data_t map(IndexArray& inds, const GradType& grad,
                            const LhsType& lhs, const RhsType& rhs) {
                return grad.eval(inds);
            }

            // template<typename LhsType, typename RhsType>
            // static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            //     return rhs.size();
            // }
        };
    };
};

struct Mul : public BinaryBasicOperator {
    template<typename LhsType, typename RhsType>
    static data_t map(IndexArray& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) * rhs.eval(inds);
    }

    struct Grad {
        using allow_broadcast = std::true_type;

        struct Lhs {
            using allow_broadcast = allow_broadcast;
            using is_lhs = std::true_type;
            using is_rhs = std::false_type;

            template<typename GradType, typename LhsType, typename RhsType>
            static data_t map(IndexArray& inds, const GradType& grad,
                            const LhsType& lhs, const RhsType& rhs) {
                return grad.eval(inds) * rhs.eval(inds);
            }

            // template<typename LhsType, typename RhsType>
            // static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            //     return lhs.size();
            // }
        };

        struct Rhs {
            using allow_broadcast = allow_broadcast;
            using is_lhs = std::false_type;
            using is_rhs = std::true_type;

            template<typename GradType, typename LhsType, typename RhsType>
            static data_t map(IndexArray& inds, const GradType& grad,
                            const LhsType& lhs, const RhsType& rhs) {
                return grad.eval(inds) * lhs.eval(inds);
            }

            // template<typename LhsType, typename RhsType>
            // static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            //     return rhs.size();
            // }
        };
    };
};

struct Sub : public BinaryBasicOperator{
    template<typename LhsType, typename RhsType>
    static data_t map(IndexArray& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) - rhs.eval(inds);
    }

    struct Grad {
        using allow_broadcast = std::true_type;

        struct Lhs {
            using allow_broadcast = allow_broadcast;
            using is_lhs = std::true_type;
            using is_rhs = std::false_type;

            template<typename GradType, typename LhsType, typename RhsType>
            static data_t map(IndexArray& inds, const GradType& grad,
                            const LhsType& lhs, const RhsType& rhs) {
                return grad.eval(inds);
            }

            // template<typename LhsType, typename RhsType>
            // static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            //     return lhs.size();
            // }
        };

        struct Rhs {
            using allow_broadcast = allow_broadcast;
            using is_lhs = std::false_type;
            using is_rhs = std::true_type;

            template<typename GradType, typename LhsType, typename RhsType>
            static data_t map(IndexArray& inds, const GradType& grad,
                            const LhsType& lhs, const RhsType& rhs) {
                return -grad.eval(inds);
            }

            // template<typename LhsType, typename RhsType>
            // static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            //     return rhs.size();
            // }
        };
    };
};

struct ReLU: public UnaryBasicOperator {
    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand) {
        return std::max(operand.eval(inds), 0.);
    }

    struct Grad {
        using allow_broadcast = std::true_type;
        using is_lhs = std::false_type;
        using is_rhs = std::false_type;

        template<typename GradType, typename OperandType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const OperandType& operand) {
            return operand.eval(inds) > 0 ? grad.eval(inds) : 0;
        }
    };
};

struct Sigmoid: public UnaryBasicOperator {
    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand) {
        return 1 / (1+std::exp(-operand.eval(inds)));
    }

    struct Grad {
        using allow_broadcast = std::true_type;
        using is_lhs = std::false_type;
        using is_rhs = std::false_type;

        template<typename GradType, typename OperandType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const OperandType& operand) {
            data_t value = Sigmoid::map(inds, operand);
            return value * (1 - value) * grad.eval(inds);
        }
    };
};

}  // namespace op
}  // namespace st
#endif