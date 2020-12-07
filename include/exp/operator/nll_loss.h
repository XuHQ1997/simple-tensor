#ifndef EXP_OPERATOR_NLL_LOSS_H
#define EXP_OPERATOR_NLL_LOSS_H

#include "utils/base_config.h"

namespace st{
namespace op {

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