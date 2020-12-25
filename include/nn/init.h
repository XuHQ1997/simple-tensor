#ifndef NN_INIT_H
#define NN_INIT_H

#include <random>
#include <string>

#include "utils/exception.h"
#include "tensor/tensor.h"
#include "tensor/tensor_impl.h"

namespace st {
namespace nn {

class InitializerBase {
public:
    InitializerBase(Tensor& param)
            : param_(const_cast<TensorImpl&>(param.impl())) {
        CHECK_TRUE(param_.is_contiguous(), 
            "Only contiguous Tensor can be initialized.");
    }

    virtual void init(void) const = 0;

protected:
    index_t data_size(void) const {
        return param_.shape_.dsize();
    }

    data_t* get_storage(void) const {
        return param_.storage_.dptr_;
    }

    static std::default_random_engine engine_;
    TensorImpl& param_;
};

class CpyInitializer : public InitializerBase {
public:
    CpyInitializer(Tensor& param, data_t* data);
    void init(void) const;
private:
    data_t* data_;
};

class KaimingInitializer : public InitializerBase {
public:
    enum Mode { fan_in = 0, fan_out};

    KaimingInitializer(Tensor& param, Mode mode=Mode::fan_in, 
                       bool conv_weight=false);
    void init(void) const;
private:
    Mode mode_;
    bool conv_weight_;
};

class UniformInitializer : public InitializerBase {
public:    
    UniformInitializer(Tensor& param, data_t a=0., data_t b=1.);
    void init(void) const;
private:
    data_t a_;
    data_t b_;
};

}  // namespace nn
}  // namespace st
#endif