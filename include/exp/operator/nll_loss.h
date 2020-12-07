#ifndef EXP_OPERATOR_NLL_LOSS_H
#define EXP_OPERATOR_NLL_LOSS_H

#include "utils/base_config.h"

namespace st{
namespace op {

struct MeanReduce {
    template<typename OperandType>
    static index_t ndim(const OperandType& operand) { return operand.ndim() - 1; }

    template<typename OperandType>
    static index_t size(index_t idx, const OperandType& operand, index_t reduce_dim) {
        if(operand.ndim() == 1) return 1;
        else if(idx < reduce_dim) return operand.size(idx);
        else return operand.size(idx + 1);
    }

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
};


struct NLLLoss {
    template<typename OperandType>
    static index_t ndim(const OperandType& operand) { return 1; }

    template<typename OperandType>
    static index_t size(index_t idx, const OperandType& operand) {
        return operand.size(0);
    }

    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand, 
                      index_t* batch_label) {
        index_t idx = inds[0];
        index_t label = batch_label[idx];
        IndexArray operand_inds{idx, label};
        return -operand.eval(operand_inds);
    }
};

}  // namespace op
}  // namespace st

#endif