#ifndef EXP_OPERATOR_CONV_H
#define EXP_OPERATOR_CONV_H

#include <utility>
#include <algorithm>
#include <iostream>

#include "utils/base_config.h"

namespace st {
namespace op {

struct Img2col {
    using Wsize = std::pair<index_t, index_t>;

    template<typename OperandType>
    static index_t ndim(const OperandType& operand) { return 2; }

    template<typename OperandType>
    static index_t size(index_t idx, const OperandType& operand, 
                        index_t n_channel, index_t n_batch,
                        const Wsize& kernel_size, const Wsize& out_size) {
        return 0;
    }

    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand,
                      index_t n_channel, index_t n_batch,
                      const Wsize& kernel_size, const Wsize& out_size) {
        return 0;
    }
};

struct MaxPool2d {
    using Wsize = std::pair<index_t, index_t>;

    template<typename OperandType>
    static index_t ndim(const OperandType& operand) { return 4; }

    template<typename OperandType>
    static index_t size(index_t idx, const OperandType& operand, 
                        const Wsize& out_size) {
        switch(idx) {
            case 1: return operand.size(1);  // num_channel
            case 2: return out_size.first;
            case 3: return out_size.second;
            default: return operand.size(0);  // num_batch
        }
    }

    template<typename OperandType>
    static data_t map(IndexArray& inds, const OperandType& operand,
                      const Wsize& kernel_size, const Wsize& stride_size,
                      const Wsize& padding_size) {
        index_t h = operand.size(2);
        index_t w = operand.size(3);
        index_t h_start = inds[2] * stride_size.first;
        index_t w_start = inds[3] * stride_size.second;
        index_t h_end = h_start + kernel_size.first;
        index_t w_end = w_start + kernel_size.second;
        IndexArray operand_inds(inds);

        data_t value, max_value = DATA_MIN;
        for(index_t i = h_start; i < h_end; ++i) {
            if(i < padding_size.first || i >= h + padding_size.first) {
                max_value = std::max(max_value, 0.);
                continue;
            }
            for(index_t j = w_start; j < w_end; ++j) {
                if(j < padding_size.second || j >= w + padding_size.second) {
                    value = 0;
                } else {
                    operand_inds[2] = i - padding_size.first;
                    operand_inds[3] = j - padding_size.second;
                    value = operand.eval(operand_inds);
                }
                max_value = std::max(max_value, value);
            }
        }

        return max_value;
    }
};

}  // namespace op
}  // namespace st


#endif