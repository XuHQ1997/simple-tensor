#include <chrono>
#include <cmath>

#include "nn/init.h"


namespace st {
namespace nn {

std::default_random_engine InitializerBase::engine_(
    std::chrono::system_clock::now().time_since_epoch().count()
);

CpyInitializer::CpyInitializer(Tensor& param, data_t* data)
        : InitializerBase(param), data_(data)
    {}

void CpyInitializer::init(void) const {
    data_t* storage_dptr = get_storage();
    index_t dsize = data_size();
    for(index_t i = 0; i < dsize; ++i)
        storage_dptr[i] = data_[i];
}

KaimingInitializer::KaimingInitializer(Tensor& param, Mode mode, 
                                       bool conv_weight)
        : InitializerBase(param), mode_(mode), 
          conv_weight_(conv_weight)
    {}

void KaimingInitializer::init(void) const {
    index_t fan;
    if((mode_ == Mode::fan_in) ^ conv_weight_)
        fan = param_.size(0);
    else
        fan = param_.size(1);

    data_t gain = std::sqrt(2.);
    data_t delta = gain * std::sqrt(fan);
    std::normal_distribution<data_t> u(0, delta);

    data_t* storage_dptr = get_storage();
    index_t dsize = data_size();

    for(index_t i = 0; i < dsize; ++i)
        storage_dptr[i] = u(engine_);
}

NormalInitializer::NormalInitializer(Tensor& param, data_t mean, data_t delta)
        : InitializerBase(param), mean_(mean), delta_(delta)
    {}

void NormalInitializer::init(void) const {
    std::normal_distribution<data_t> u(mean_, delta_);

    data_t* storage_dptr = get_storage();
    index_t dsize = data_size();
    for(index_t i = 0; i < dsize; ++i)
        storage_dptr[i] = u(engine_);
}

}  // namespace nn
}  // namespace st