#include <iostream>
#include <chrono>

#include "utils/base_config.h"
#include "utils/allocator.h"
#include "exp/function.h"
#include "tensor/tensor.h"
#include "nn/module.h"
#include "data/data.h"
#include "nn/optim.h"

using st::index_t;
using st::data_t;


class SimpleCNN : public st::nn::Module {
public:
    SimpleCNN() = default;
    ~SimpleCNN() = default;

    st::Tensor forward(const st::Tensor& input) {
        st::Tensor s0_x1 = conv0.forward(input);

        st::Tensor s1_x1 = s1_conv1.forward(s0_x1);
        st::Tensor s1_x2 = s1_conv2.forward(s1_x1);
        st::Tensor s1_x3 = s1_pool.forward(s1_x2);

        st::Tensor s2_x1 = s2_conv1.forward(s1_x3);
        st::Tensor s2_x2 = s2_conv2.forward(s2_x1);
        st::Tensor s2_x3 = s2_pool.forward(s2_x2);

        st::Tensor feat(s2_x3.size());
        feat = s2_x3;

        st::Tensor y1 = linear1.forward(feat.view({
            feat.size(0), 32*4*4
        }));
        st::Tensor y2 = linear2.forward(y1);
        return y2;
    }

    st::nn::ParamsDict parameters(void) {
        return {
            {"conv0", conv0.parameters()},
            {"s1_conv1", s1_conv1.parameters()},
            {"s1_conv2", s1_conv2.parameters()},
            {"s2_conv1", s2_conv1.parameters()},
            {"s2_conv2", s2_conv2.parameters()},
            {"linear1", linear1.parameters()},
            {"linear2", linear2.parameters()}
        };
    }
private:
    st::nn::Conv2dWithReLU conv0{3, 16, {5, 5}, {2, 2}, {2, 2}};

    st::nn::Conv2dWithReLU s1_conv1{16, 16, {3, 3}, {1, 1}, {1, 1}};
    st::nn::Conv2dWithReLU s1_conv2{16, 16, {3, 3}, {1, 1}, {1, 1}};
    st::nn::MaxPool2d s1_pool{{2, 2}, {2, 2}, {0, 0}};

    st::nn::Conv2dWithReLU s2_conv1{16, 32, {3, 3}, {1, 1}, {1, 1}};
    st::nn::Conv2dWithReLU s2_conv2{32, 32, {3, 3}, {1, 1}, {1, 1}};
    st::nn::MaxPool2d s2_pool{{2, 2}, {2, 2}, {0, 0}};

    st::nn::LinearWithReLU linear1{32*4*4, 256};
    st::nn::Linear linear2{256, 10};
};

int main() {
    // config
    constexpr index_t epoch = 4;
    constexpr index_t batch_size = 64;
    constexpr data_t lr = 0.01;
    constexpr data_t momentum = 0.9;
    constexpr index_t print_iters = 10;

    using namespace std::chrono;
    steady_clock::time_point start_tp = steady_clock::now();

    // dataset
    st::data::Cifar10 train_dataset(
        /*dataset_dir=*/"D:\\storehouse\\dataset\\cifar-10-batches-bin",
        /*train=*/true,
        /*batch_size=*/batch_size,
        /*shuffle=*/false,
        /*path_sep=*/'\\'
    );
    st::data::Cifar10 val_dataset(
        /*dataset_dir=*/"D:\\storehouse\\dataset\\cifar-10-batches-bin",
        /*train=*/false,
        /*batch_size=*/batch_size,
        /*shuffle=*/false,
        /*path_sep=*/'\\'
    );
    std::cout << "train dataset length: " << train_dataset.n_samples() << std::endl;
    std::cout << "val dataset length: " << val_dataset.n_samples() << std::endl;

    // model and criterion
    SimpleCNN scnn;
    st::nn::CrossEntropy criterion;

    // optimizer
    st::nn::SGDwithMomentum optimizer(
        scnn.parameters(), /*lr=*/lr, /*momentum=*/momentum
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
                {n_samples, 
                 st::data::Cifar10::Img::n_channels_, 
                 st::data::Cifar10::Img::n_rows_, 
                 st::data::Cifar10::Img::n_cols_}
            );

            st::Tensor output = scnn.forward(input);
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
                {n_samples,
                 st::data::Cifar10::Img::n_channels_, 
                 st::data::Cifar10::Img::n_rows_, 
                 st::data::Cifar10::Img::n_cols_}
            );

            st::Tensor output = scnn.forward(input);
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
