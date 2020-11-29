#ifndef UTILS_BASE_CONFIG_H
#define UTILS_BASE_CONFIG_H

namespace st {

using index_t = unsigned int;
using data_t = double;

template<typename Dtype> class DynamicArray;
using IndexArray = DynamicArray<index_t>;

} // namespace st
#endif