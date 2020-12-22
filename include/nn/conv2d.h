#ifndef NN_CONV2D_H
#define NN_CONV2D_H

#include "utils/base_config.h"
#include "exp/operator/conv.h"
#include "exp/function.h"
#include "nn/nn.h"

namespace st {
namespace nn {

class Conv2d : public Module {
public:
    using Wsize = op::Img2col::Wsize;

    // Conv2d(index_t in_channels, index_t out_channels, 
    //        const Wsize& kernel_size, const Wsize& stride,
    //        const Wsize& padding)
    //         : kernel_size_(kernel_size), 
    //           stride_(stride),
    //           padding_(padding),
    //           weight_("weight", Tensor(Shape{
    //               out_channels, in_channels*kernel_size.first*kernel_size.second})),
    //         //   bias_("bias", Tensor(Shape{op::Img2col::Wsize}))
    //     {}

    // data member
    Wsize kernel_size_;
    Wsize stride_;
    Wsize padding_;

    // output = img2col(img) @ weight.T + bias
    // img2col(img) : [oh*ow*b, c*kh*kw]
    // weight: [oc, c*kh*kw]
    Params weight_;
    Params bias_;
};
}  // namespace nn
}  // namespace st
#endif