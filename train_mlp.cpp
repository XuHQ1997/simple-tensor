#include <iostream>
#include <chrono>

// The next line will cancel CHECK_XXX macro used in header files,
// but have no effect on these in src files.
#define CANCEL_CHECK

#include "utils/base_config.h"
#include "utils/allocator.h"
#include "exp/function.h"
#include "tensor/tensor.h"
#include "nn/module.h"
#include "data/mnist.h"
#include "nn/optim.h"
#include "nn/init.h"

using st::index_t;
using st::data_t;

class MLP : public st::nn::Module {
public:
    MLP(index_t in, index_t hidden1, index_t hidden2, index_t out)
            : linear1_(in, hidden1),
              linear2_(hidden1, hidden2),
              linear3_(hidden2, out)
        {}
    ~MLP() = default;

    st::Tensor forward(const st::Tensor& input) {
        st::Tensor x1 = linear1_.forward(input);
        st::Tensor x2 = linear2_.forward(x1);
        st::Tensor y = linear3_.forward(x2);
        return y;
    }

    st::nn::ParamsDict parameters(void) {
        return {
            {"linear1", linear1_.parameters()},
            {"linear2", linear2_.parameters()},
            {"linear3", linear3_.parameters()}
        };
    }
private:
    st::nn::LinearWithReLU linear1_;
    st::nn::LinearWithReLU linear2_;
    st::nn::Linear linear3_;
};


int main() {
    // config
    constexpr index_t epoch = 2;
    constexpr index_t batch_size = 64;
    constexpr data_t lr = 0.05;
    constexpr data_t momentum = 0.9;
    constexpr index_t print_iters = 10;

    using namespace std::chrono;
    steady_clock::time_point start_tp = steady_clock::now();

    // dataset
    st::data::MNIST train_dataset(
        /*img_path=*/"D:\\storehouse\\dataset\\MNIST\\train-images.idx3-ubyte",
        /*label_path=*/"D:\\storehouse\\dataset\\MNIST\\train-labels.idx1-ubyte",
        /*batch_size=*/batch_size,
        /*shuffle=*/false
    );
    st::data::MNIST val_dataset(
        /*img_path=*/"D:\\storehouse\\dataset\\MNIST\\t10k-images.idx3-ubyte",
        /*label_path=*/"D:\\storehouse\\dataset\\MNIST\\t10k-labels.idx1-ubyte",
        /*batch_size=*/batch_size,
        /*shuffle=*/false
    );

    // model and criterion
    MLP mlp(st::data::MNIST::Img::n_pixels_, 512, 512, 10);
    st::nn::CrossEntropy criterion;

    // optimizer
    st::nn::SGDwithMomentum optimizer(
        mlp.parameters(), /*lr=*/lr, /*momentum=*/momentum
    );

    index_t n_samples;
    const data_t* batch_samples;
    const index_t* batch_labels;
    for(index_t i = 0; i < epoch; ++i) {
        std::cout << "Epoch " << i << " training..." << std::endl;
        std::cout << "total iters: " << train_dataset.n_batchs() << std::endl;
        train_dataset.shuffle();
        for(index_t j = 0; j < train_dataset.n_batchs(); ++j) {
            std::tie(n_samples, batch_samples, batch_labels) = 
                train_dataset.get_batch(j);
            st::Tensor input(
                batch_samples, 
                {n_samples, st::data::MNIST::Img::n_pixels_}
            );

            st::Tensor output = mlp.forward(input);
            st::Tensor loss = criterion.forward(output, batch_labels);
            loss.backward();

            optimizer.step();
            optimizer.zero_grad();

            if(j % print_iters == 0) {
                std::cout << "iter " << j << " | ";
                std::cout << "loss: " << loss.item() << std::endl;
            }
        }

        std::cout << "Epoch " << i << " evaluating..." << std::endl;
        index_t total_samples = 0, correct_samples = 0;
        for(index_t j = 0; j < val_dataset.n_batchs(); ++j) {
            std::tie(n_samples, batch_samples, batch_labels) =
                val_dataset.get_batch(j);
            st::Tensor input(
                batch_samples,
                {n_samples, st::data::MNIST::Img::n_pixels_}
            );

            st::Tensor output = mlp.forward(input);
            st::Tensor predict = st::op::argmax(output, 1);
            for(index_t k = 0; k < n_samples; ++k) {
                ++total_samples;
                index_t pd_label = predict[{k}];
                if(pd_label == batch_labels[k])
                    ++correct_samples;
            }
        }
        std::cout << "total samples: " << total_samples;
        std::cout << " | correct samples: " << correct_samples;
        std::cout << " | acc: ";
        std::cout << static_cast<data_t>(correct_samples) / total_samples << std::endl;
    }

    steady_clock::time_point end_tp = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(end_tp - start_tp);
    std::cout << "Training finished. Training took " << time_span.count();
    std::cout << " seconds." << std::endl;
    return 0;
}
