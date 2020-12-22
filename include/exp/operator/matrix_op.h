#ifndef EXP_OPERATOR_MATRIX_MUL_H
#define EXP_OPERATOR_MATRIX_MUL_H

#include <algorithm>

#include "utils/base_config.h"


namespace st {
namespace op {

struct MatrixTranspose {
    
    template<typename OperandType>
    static index_t ndim(const OperandType& operand) { return 2; }

    template<typename OperandType>
    static index_t size(index_t idx, const OperandType& operand) {
        return operand.size(1 - idx);
    }

    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand) {
        std::swap(inds[0], inds[1]);
        data_t value = operand.eval(inds);
        std::swap(inds[0], inds[1]);
        return value;
    }

    struct Grad {
        template<typename GradType, typename OperandType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const OperandType& operand) {
            std::swap(inds[0], inds[1]);
            data_t value = grad.eval(inds);
            std::swap(inds[0], inds[1]);
            return value;
        }
    };
};

struct MatrixMul {
    
    template<typename LhsType, typename RhsType>
    static index_t ndim(const LhsType& lhs, const RhsType& rhs) { return 2; }

    template<typename LhsType, typename RhsType>
    static index_t size(index_t idx, const LhsType& lhs, const RhsType& rhs) {
        return idx == 0 ? lhs.size(0) : rhs.size(1);
    }

    template<typename LhsType, typename RhsType>
    static data_t map(IndexArray& inds, const LhsType& lhs, const RhsType& rhs) {
        index_t hsize = lhs.size(1);
        IndexArray lhs_inds(inds);
        IndexArray rhs_inds(inds);

        data_t value = 0;
        for(index_t i = 0; i < hsize; ++i) {
            lhs_inds[1] = i;
            rhs_inds[0] = i;
            value += lhs.eval(lhs_inds) * rhs.eval(rhs_inds);
        }
        return value;
    }

    struct LhsGrad {
        template<typename GradType, typename LhsType, typename RhsType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const LhsType& lhs, const RhsType& rhs) {
            index_t hsize = rhs.size(1);
            IndexArray grad_inds({inds[0], 0});
            IndexArray rhs_inds({inds[1], 0});

            data_t value = 0;
            for(index_t i = 0; i < hsize; ++i) {
                grad_inds[1] = i;
                rhs_inds[1] = i;
                value += grad.eval(grad_inds) * rhs.eval(rhs_inds);
            }
            return value;
        }

        template<typename LhsType, typename RhsType>
        static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            return lhs.size();
        }
    };

    struct RhsGrad {
        template<typename GradType, typename LhsType, typename RhsType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const LhsType& lhs, const RhsType& rhs) {
            index_t hsize = lhs.size(0);
            IndexArray lhs_inds({0, inds[0]});
            IndexArray grad_inds({0, inds[1]});

            data_t value = 0;
            for(index_t i = 0; i < hsize; ++i) {
                lhs_inds[0] = i;
                grad_inds[0] = i;
                value += lhs.eval(lhs_inds) * grad.eval(grad_inds);
            }
            return value;
        }

        template<typename LhsType, typename RhsType>
        static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            return rhs.size();
        }
    };
};

struct BatchMatrixTranspose {
    
    template<typename OperandType>
    static index_t ndim(const OperandType& operand) { return 3; }

    template<typename OperandType>
    static index_t size(index_t idx, const OperandType& operand) {
        switch(idx) {
            case 0: return operand.size(0);
            case 1: return operand.size(2);
            default: return operand.size(1); // case 2
        }
    }

    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand) {
        std::swap(inds[1], inds[2]);
        return operand.eval(inds);
    }

    struct Grad {
        template<typename GradType, typename OperandType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const OperandType& operand) {
            std::swap(inds[1], inds[2]);
            return grad.eval(inds);
        }
    };
};

struct BatchMatrixMul {
    template<typename LhsType, typename RhsType>
    static index_t ndim(const LhsType& lhs, const RhsType& rhs) { return 3; }

    template<typename LhsType, typename RhsType>
    static index_t size(index_t idx, const LhsType& lhs, const RhsType& rhs) {
        switch(idx) {
            case 0: return std::max(lhs.size(0), rhs.size(0));
            case 1: return lhs.size(1);
            default: return rhs.size(2);  // case 2
        }
        return -1;
    }

    template<typename LhsType, typename RhsType>
    static data_t map(IndexArray& inds, const LhsType& lhs, const RhsType& rhs) {
        index_t hsize = lhs.size(2);
        IndexArray lhs_inds(inds);
        IndexArray rhs_inds(inds);

        data_t value = 0;
        for(index_t i = 0; i < hsize; ++i) {
            lhs_inds[2] = i;
            rhs_inds[1] = i;
            value += lhs.eval(lhs_inds) * rhs.eval(rhs_inds);
        }
        return value;
    }

    struct LhsGrad {
        template<typename GradType, typename LhsType, typename RhsType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const LhsType& lhs, const RhsType& rhs) {
            index_t hsize = rhs.size(2);
            IndexArray grad_inds({inds[0], inds[1], 0});
            IndexArray rhs_inds({inds[0], inds[2], 0});

            data_t value;
            for(index_t i = 0; i < hsize; ++i) {
                grad_inds[2] = i;
                rhs_inds[2] = i;
                value += grad.eval(grad_inds) * rhs.eval(rhs_inds);
            }
            return value;
        }

        template<typename LhsType, typename RhsType>
        static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            return lhs.size();
        }
    };

    struct RhsGrad {
        template<typename GradType, typename LhsType, typename RhsType>
        static data_t map(IndexArray& inds, const GradType& grad,
                          const LhsType& lhs, const RhsType& rhs) {
            index_t hsize = lhs.size(1);
            IndexArray lhs_inds({inds[0], 0, inds[1]});
            IndexArray grad_inds({inds[0], 0, inds[2]});

            data_t value = 0;
            for(index_t i = 0; i < hsize; ++i) {
                lhs_inds[1] = i;
                grad_inds[1] = i;
                value += lhs.eval(lhs_inds) * grad.eval(grad_inds);
            }
            return value;
        }

        template<typename LhsType, typename RhsType>
        static IndexArray size(const LhsType& lhs, const RhsType& rhs) {
            return rhs.size();
        }
    };
};

}  // namespace op
}  // namespace st
#endif