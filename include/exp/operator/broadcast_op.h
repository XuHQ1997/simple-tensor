#ifndef EXP_OPERATION_BROADCAST_H
#define EXP_OPERATION_BROADCAST_H

#include <type_traits>
#include <algorithm>

#include "utils/base_config.h"


namespace st {
namespace op {

// The only difference between broadcast operator and basic operator is whether
// it can be element-wisely evaluated.
struct BroadcastOperator {
    using is_elementwise = std::false_type;
    template<typename LhsType, typename RhsType>
    static index_t ndim(const LhsType& lhs, const RhsType& rhs) { 
        return std::max(lhs.ndim(), rhs.ndim());
    }
    template<typename LhsType, typename RhsType>
    static index_t size(index_t idx, const LhsType& lhs, const RhsType& rhs) {
        return std::max(lhs.size(idx), rhs.size(idx));
    }
};

struct BroadcastAdd : public BroadcastOperator {
    template<typename LhsType, typename RhsType>
    static data_t map(IndexArray& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) + rhs.eval(inds);
    }
};

struct BroadcastMul : public BroadcastOperator {
    template<typename LhsType, typename RhsType>
    static data_t map(IndexArray& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) * rhs.eval(inds);
    }
};

struct BroadcastSub : public BroadcastOperator {
    template<typename LhsType, typename RhsType>
    static data_t map(IndexArray& inds, const LhsType& lhs, const RhsType& rhs) {
        return lhs.eval(inds) - rhs.eval(inds);
    }
};

}  // namespace op
}  // namespace st

#endif