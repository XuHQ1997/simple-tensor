#include "exp/function.h"
#include "nn/module.h"

namespace st {
namespace nn {
Linear::Linear(index_t in_features, index_t out_features)
        : weight_(Shape{out_features, in_features}, true),
          bias_(Shape{ 1, out_features}, true)
    {}

Tensor Linear::forward(const Tensor& x) {
    Tensor y1 = op::matrix_mul(
        x, op::matrix_transpose(weight_)
    );
    Tensor y2 = y1 + bias_;
    return y2;
}

ParamsDict Linear::parameters(void) {
    return {
        {"weight", weight_},
        {"bias", bias_}
    };
}

LinearWithReLU::LinearWithReLU(index_t in_features, index_t out_features)
        : Linear(in_features, out_features)
    {}

Tensor LinearWithReLU::forward(const Tensor& x) {
    Tensor y1 = op::matrix_mul(
        x, op::matrix_transpose(weight_)
    );
    Tensor y2 = op::relu(y1 + bias_);
    return y2;
}
}  // namespace nn
}  // namespace st

namespace st {
namespace nn {
Conv2d::Conv2d(index_t in_channels, index_t out_channels,
               const Wsize& kernel_size, const Wsize& stride,
               const Wsize& padding)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          weight_(Shape{
              out_channels_,
              in_channels_ * kernel_size_.first * kernel_size_.second},
              /*requires_grad=*/true)
    {}

Tensor Conv2d::forward(const Tensor& x) {
    auto col_exp = op::img2col(
        x, kernel_size_, stride_, padding_
    );

    Tensor y1 = op::matrix_mul(
        Tensor(col_exp), op::matrix_transpose(weight_)
    );

    auto&& conv_feat_size = col_exp.impl().conv_feat_size();
    Tensor y2 = y1.view({
        conv_feat_size.first, conv_feat_size.second, x.size(0), out_channels_
    });
    Tensor y3 = y2.permute({2, 3, 0, 1});
    return y3;
}

ParamsDict Conv2d::parameters(void) {
    return {{"weight", weight_}};
}

Conv2dWithReLU::Conv2dWithReLU(index_t in_channels, index_t out_channels,
                               const Wsize& kernel_size, const Wsize& stride,
                               const Wsize& padding)
        : Conv2d(in_channels, out_channels, 
                 kernel_size, stride, padding)
    {}

Tensor Conv2dWithReLU::forward(const Tensor& x) {
    auto col_exp = op::img2col(
        x, kernel_size_, stride_, padding_
    );

    Tensor y1 = op::relu(op::matrix_mul(
        Tensor(col_exp), op::matrix_transpose(weight_)
    ));

    auto& conv_feat_size = col_exp.impl().conv_feat_size();
    Tensor y2 = y1.view({
        conv_feat_size.first, conv_feat_size.second, 
        x.size(0), out_channels_
    });
    Tensor y3 = y2.permute({2, 3, 0, 1});
    return y3;
}
}  // namespace nn
}  // namespace st

namespace st {
namespace nn {

MaxPool2d::MaxPool2d(const Wsize& kernel_size, const Wsize& stride,
                     const Wsize& padding)
        : kernel_size_(kernel_size),
          stride_(stride),
          padding_(padding)
    {}

Tensor MaxPool2d::forward(const Tensor& x) {
    auto col_exp = op::img2col(
        x, kernel_size_, stride_, padding_
    );
    Tensor y1 = col_exp;

    auto& conv_feat_size = col_exp.impl().conv_feat_size();
    Tensor y2 = y1.view({
        conv_feat_size.first, conv_feat_size.second, 
        x.size(0), x.size(1), 
        kernel_size_.first*kernel_size_.second
    });
    Tensor y3 = op::max(y2, /*dim=*/4);
    Tensor y4 = y3.permute({2, 3, 0, 1});
    return y4;
}

ParamsDict MaxPool2d::parameters(void) {
    return {};
}

Tensor CrossEntropy::forward(const Tensor& input,
                             std::shared_ptr<index_t> labels) {
    auto logits = op::log_softmax(input);
    auto nll = op::nll_loss(logits, labels);
    Tensor loss = op::mean(nll, 0);
    return loss;
}
}  // namespace nn
}  // namespace st