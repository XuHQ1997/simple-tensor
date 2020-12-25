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
    index_t fan = mode_ == Mode::fan_in ? param_.size(1)
                                        : param_.size(0);
    data_t gain = std::sqrt(2.);
    data_t bound = gain * std::sqrt(3. / fan);
    std::uniform_real_distribution<data_t> u(-bound, bound);

    data_t* storage_dptr = get_storage();
    index_t dsize = data_size();

    for(index_t i = 0; i < dsize; ++i) {
        storage_dptr[i] = u(engine_);
    }
}

UniformInitializer::UniformInitializer(Tensor& param, data_t a, data_t b)
        : InitializerBase(param), a_(a), b_(b)
    {}

void UniformInitializer::init(void) const {
    std::uniform_real_distribution<data_t> u(a_, b_);

    data_t* storage_dptr = get_storage();
    index_t dsize = data_size();
    for(index_t i = 0; i < dsize; ++i)
        storage_dptr[i] = u(engine_);
}

}  // namespace nn
}  // namespace st