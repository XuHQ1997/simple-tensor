#include <cstring>

#include "tensor/storage.h"
#include "tensor/tensor.h"
#include "tensor/tensor_impl.h"
#include "nn/optim.h"

namespace st {
namespace nn {

OptimizerBase::OptimizerBase(const ParamsDict& params_dict) {
    params_.reserve(params_dict.size());
    for(auto named_param_ref: params_dict) {
        Tensor& tensor = named_param_ref.second.get();
        TensorImpl& impl = const_cast<TensorImpl&>(tensor.impl());
        CHECK_TRUE(impl.is_contiguous(),
            "Only contiguous Tensor can be optimized.");
        params_.emplace_back(impl);
    }
}

void OptimizerBase::zero_grad(void) {
    for(TensorImpl& t: params_) {
        data_t* grad_dptr = get_grad(t);
        std::memset(grad_dptr, 0, t.shape_.dsize());
    }
}

SGD::SGD(const ParamsDict& params_dict, data_t lr)
        : OptimizerBase(params_dict), lr_(lr)
    {}

void SGD::step(void) {
    for(TensorImpl& t : params_) {
        data_t* storage_dptr = get_storage(t);
        data_t* grad_dptr = get_grad(t);
        index_t dsize = data_size(t);

        for(index_t i = 0; i < dsize; ++i)
            storage_dptr[i] -= lr_ * grad_dptr[i];
    }
}

SGDwithMomentum::SGDwithMomentum(const ParamsDict& params_dict, 
                                 data_t lr, data_t momentum)
        : OptimizerBase(params_dict),
          lr_(lr), momentum_(momentum) {
    running_means_.reserve(params_.size());
    for(TensorImpl& t : params_)
        running_means_.emplace_back(
            Alloc::shared_allocate<data_t>(
                sizeof(data_t) * data_size(t)
            )
        );
}

void SGDwithMomentum::step(void) {
    for(index_t i = 0; i < params_.size(); ++i) {
        TensorImpl& t = params_[i];
        data_t* storage_dptr = get_storage(t);
        data_t* grad_dptr = get_grad(t);
        data_t* vx = running_means_[i].get();
        index_t dsize = data_size(t);

        for(index_t j = 0; j < dsize; ++j) {
            vx[i] = momentum_ * vx[i] + grad_dptr[i];
            storage_dptr[i] -= lr_ * vx[i];
        }
    }
}


}  // namespace nn
}  // namespace st