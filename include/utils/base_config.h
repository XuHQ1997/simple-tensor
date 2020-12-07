#ifndef UTILS_BASE_CONFIG_H
#define UTILS_BASE_CONFIG_H

#include <limits>

namespace st {

using index_t = unsigned int;
using data_t = double;

template<typename Dtype> class DynamicArray;
using IndexArray = DynamicArray<index_t>;

// INDEX_MAX =  UINT_MAX/2, to check negative value.
// See CHECK_INDEX_INVALID in utils/exception.h for the reason. 
constexpr index_t INDEX_MAX = std::numeric_limits<index_t>::max() >> 1;
constexpr index_t INDEX_MIN = 0;
constexpr data_t DATA_MAX = std::numeric_limits<data_t>::max();
constexpr data_t DATA_MIN = std::numeric_limits<data_t>::min();

} // namespace st
#endif