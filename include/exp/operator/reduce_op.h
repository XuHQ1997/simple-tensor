#ifndef EXP_OPERATOR_REDUCE_OP_H
#define EXP_OPERATOR_REDUCE_OP_H

#include <algorithm>

#include "utils/base_config.h"
#include "utils/exception.h"

namespace st {
namespace op {

struct ReduceOperator {
    template<typename OperandType>
    static index_t ndim(const OperandType& operand) { 
        return std::max(operand.ndim() - 1, static_cast<index_t>(1)); 
    }

    template<typename OperandType>
    static index_t size(index_t idx, const OperandType& operand, index_t reduce_dim) {
        if(operand.ndim() == 1) return 1;
        else if(idx < reduce_dim) return operand.size(idx);
        else return operand.size(idx + 1);
    }
};

struct MeanReduce : public ReduceOperator {
    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand, 
                      index_t reduce_dim) {
        IndexArray operand_inds(inds.size() + 1);
        index_t reduce_size = operand.size(reduce_dim);
        index_t i = 0;
        for(; i < reduce_dim; ++i)  operand_inds[i] = inds[i];
        for(++i; i < inds.size() + 1; ++i) operand_inds[i] = inds[i-1];
        
        data_t value = 0;
        for(index_t i = 0; i < reduce_size; ++i) {
            operand_inds[reduce_dim] = i;
            value += operand.eval(operand_inds);
        }
        value /= reduce_size;
        return value;
    }

    struct Grad {    
        template<typename GradType, typename OperandType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const OperandType& operand, 
                          index_t reduce_dim) {
            index_t reduce_size = operand.size(reduce_dim);
            return grad.eval(inds) / reduce_size;
        }
    };
};

struct Argmax : public ReduceOperator {
    template<typename OperandType>
    static index_t map(IndexArray& inds, const OperandType& operand, 
                      index_t reduce_dim) {
        IndexArray operand_inds(inds.size() + 1);
        index_t reduce_size = operand.size(reduce_dim);
        index_t i = 0;
        for(; i < reduce_dim; ++i)  operand_inds[i] = inds[i];
        for(++i; i < inds.size() + 1; ++i) operand_inds[i] = inds[i-1];

        data_t value, max_value = DATA_MIN;
        index_t idx;
        for(i = 0; i < reduce_size; ++i) {
            operand_inds[reduce_dim] = i;
            value = operand.eval(operand_inds);
            if(max_value < value) {
                max_value = value;
                idx = i;
            }
        }
        return idx;
    }

    struct Grad {
        template<typename GradType, typename OperandType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const OperandType& operand, 
                          index_t reduce_dim) {
            index_t reduce_size = operand.size(reduce_dim);
            data_t key_idx = inds[reduce_dim];
            data_t key_value = operand.eval(inds);
            data_t value = DATA_MIN;

            index_t i = 0;
            for(; i < key_idx && value < key_value; ++i) {
                inds[reduce_dim] = i;
                value = operand.eval(inds);
            }
            if(i != key_idx) return 0;

            for(++i; i < reduce_size && value < key_value; ++i) {
                inds[reduce_dim] = i;
                value = operand.eval(inds);
            }
            if(i != reduce_size) return 0;

            IndexArray grad_inds(inds.size() - 1);
            for(i = 0; i < reduce_dim; ++i) grad_inds[i] = inds[i];
            for(++i; i < reduce_size; ++i) grad_inds[i-1] = inds[i];
            return grad.eval(grad_inds);
        }
    };
};

}  // namespace op
}  // namespace st

#endif