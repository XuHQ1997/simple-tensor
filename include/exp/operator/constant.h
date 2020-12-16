#ifndef EXP_OPERATOR_CONSTANT_H
#define EXP_OPERATOR_CONSTANT_H

#include "utils/base_config.h"
#include "utils/exception.h"

namespace st {
namespace op {


struct Constant {
    static index_t ndim() { return 1; }
    static index_t size(index_t idx) { return 1; }
    static data_t map(IndexArray& inds, data_t value) {
        return value;
    }

    struct Grad {
        static data_t map(IndexArray& inds, data_t value) {
            THROW_ERROR("NotImplementError");
            return 0;
        }
    };
};


}  // namespace op
}  // namespace st
#endif