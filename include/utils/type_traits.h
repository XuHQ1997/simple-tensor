#ifndef UTILS_TYPE_TRAITS_H
#define UTILS_TYPE_TRAITS_H

#include <type_traits>

// std::disconjunction isn't supported until c++17.
namespace std {

template<typename B1, typename... Bn>
struct logic_and : std::conditional<bool(B1::value), logic_and<Bn...>, B1>::type {};

template<typename B1> struct logic_and<B1> : B1 {};

template<typename B1, typename... Bn>
using logic_and_t = typename logic_and<B1, Bn...>::type;

}  // namespace st

#endif
