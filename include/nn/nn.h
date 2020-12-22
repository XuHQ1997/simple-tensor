#ifndef NN_NN_H

#include <string>
#include <map>

#include "tensor/tensor.h"

namespace st {
namespace nn {

using Params = std::pair<std::string, Tensor>;
using ParamsDict = std::map<std::string, Tensor>;

class Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual ParamsDict parameters(void) = 0;
    virtual ~Module() = default;
};

}  // namespace nn
}  // namespace st
#endif