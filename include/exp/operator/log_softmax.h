#ifndef EXP_OPERATOR_LOG_SOFTMAX_H
#define EXP_OPERATOR_LOG_SOFTMAX_H

#include <cmath>
#include <memory>

#include "utils/base_config.h"

namespace st {
namespace op {

// This operator need specialize UnaryExpImpl in exp/exp_impl.h
struct LogSoftmax {

    template<typename OperandType>
    static index_t ndim(const OperandType& operand) { return 2; }

    template<typename OperandType>
    static index_t size(index_t idx, const OperandType& operand) {
        return operand.size(idx);
    }

    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand, 
                      data_t* batch_sum_exp, data_t* batch_max_cls) {
        data_t value = operand.eval(inds);
        return value - batch_max_cls[inds[0]] - std::log(batch_sum_exp[inds[0]]);
    }

    template<typename OperandType>
    static void precompute(const OperandType& operand, data_t* batch_sum_exp,
                           data_t* batch_max_cls) {
        index_t n_batch = operand.size(0);
        index_t n_class = operand.size(1);
        auto batch_ptr = Alloc::shared_allocate<data_t>(n_class * sizeof(data_t));
        auto batch = batch_ptr.get();
        IndexArray inds(2);

        for(index_t i = 0; i < n_batch; ++i) {
            inds[0] = i;
            data_t max_cls = DATA_MIN;
            for(index_t j = 0; j < n_class; ++j) {
                inds[1] = j;
                batch[j] = operand.eval(inds);
                if(batch[j] > max_cls)
                    max_cls = batch[j];
            }

            data_t sum_exp = 0;
            for(int j = 0; j < n_class; ++j)
                sum_exp += std::exp(batch[j] - max_cls);

            batch_sum_exp[i] = sum_exp;
            batch_max_cls[i] = max_cls;
        }
    }

    struct Grad {
        template<typename GradType, typename OperandType>
        static data_t map(IndexArray& inds, const GradType& grad, 
                          const OperandType& operand, data_t* batch_sum_exp,
                          data_t* batch_max_cls) {
            data_t x = operand.eval(inds);
            data_t softmax = (std::exp(x - batch_max_cls[inds[0]])) / batch_sum_exp[inds[0]];

            index_t n_cls = operand.size(1);
            IndexArray grad_inds(inds);

            data_t total_grad = 0;
            index_t i = 0;
            for(; i < inds[1]; ++i) {
                grad_inds[1] = i;
                total_grad -= grad.eval(grad_inds);
            }
            for(++i; i < n_cls; ++i) {
                grad_inds[1] = i;
                total_grad -= grad.eval(grad_inds);
            }
            total_grad *= softmax;

            grad_inds[1] = inds[1];
            total_grad += (1 - softmax) * grad.eval(grad_inds);
            return total_grad;
        }
    };
};



}  // namespace op
}  // namespace st
#endif