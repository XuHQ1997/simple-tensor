#ifndef UTILS_BASE_CONFIG_H
#define UTILS_BASE_CONFIG_H

#include <limits>

namespace st {

using index_t = unsigned int;
using data_t = double;

template<typename Dtype> class DynamicArray;
using IndexArray = DynamicArray<index_t>;

constexpr data_t DATA_MAX = std::numeric_limits<data_t>::max();
constexpr data_t DATA_MIN = std::numeric_limits<data_t>::min();

} // namespace st
#endif