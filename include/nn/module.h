#ifndef NN_MODULE_H
#define NN_MODULE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <initializer_list>

#include "tensor/tensor.h"

namespace st {
namespace nn {


class ParamsDict 
        : public std::unordered_map<std::string, std::reference_wrapper<Tensor>> {
public:
    ParamsDict() = default;
    ParamsDict(std::initializer_list<value_type> items)
            : std::unordered_map<std::string, std::reference_wrapper<Tensor>>(items)
        {}
    ParamsDict(std::initializer_list<std::pair<std::string, ParamsDict>> dicts) {
        for(auto& named_dict: dicts) {
            auto& name = named_dict.first;
            auto& dict = named_dict.second;

            for(auto iter: dict)
                this->insert({
                    name + iter.first,
                    iter.second
                });
        }
    }

    Tensor& operator[](const std::string& key) {
        auto iter = find(key);
        return iter->second.get();
    }
};

class Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual ParamsDict parameters(void) = 0;
    virtual ~Module() = default;
};

class Linear : public Module {
public:
    Linear(index_t in_features, index_t out_features);
    Linear(const Linear& other) = delete;
    ~Linear() = default;

    Tensor forward(const Tensor& input) override;
    ParamsDict parameters(void) override;
protected:
    Tensor weight_;
    Tensor bias_;
};

class LinearWithReLU : public Linear {
public:
    LinearWithReLU(index_t in_features, index_t out_features);
    Tensor forward(const Tensor& input) override;
};

class Conv2d : public Module {
public:
    using Wsize = op::Img2col::Wsize;

    Conv2d(index_t in_channels, index_t out_channels, 
           const Wsize& kernel_size, const Wsize& stride, 
           const Wsize& padding);
    Conv2d(const Conv2d& other) = delete;
    ~Conv2d() = default;

    Tensor forward(const Tensor& input) override;
    ParamsDict parameters(void) override;
protected:
    index_t in_channels_;
    index_t out_channels_;

    Wsize kernel_size_;
    Wsize stride_;
    Wsize padding_;

    Tensor weight_;
};

class Conv2dWithReLU : public Conv2d {
public:
    Conv2dWithReLU(index_t in_channels, index_t out_channels,
                   const Wsize& kernel_size, const Wsize& stride,
                   const Wsize& padding);
    Tensor forward(const Tensor& input) override;
};

class MaxPool2d : public Module {
public:
    using Wsize = op::Img2col::Wsize;

    MaxPool2d(const Wsize& kernel_size, const Wsize& stride, 
              const Wsize& padding);
    MaxPool2d(const MaxPool2d& other) = delete;
    ~MaxPool2d() = default;

    Tensor forward(const Tensor& input) override;
    ParamsDict parameters(void) override;
protected:
    Wsize kernel_size_;
    Wsize stride_;
    Wsize padding_;
};

class CrossEntropy {
public:
    CrossEntropy() = default;
    ~CrossEntropy() = default;

    Tensor forward(const Tensor& input,
                   const index_t* labels);
};

}  // namespace nn
}  // namespace st
#endif