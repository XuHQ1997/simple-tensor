#include <fstream>
#include <memory>
#include <vector>
#include <cstring>
#include <random>
#include <chrono>
#include <algorithm>

#include "data/mnist.h"

namespace st {
namespace data {

unsigned int __reverse_int(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

MNIST::MNIST(const std::string& img_path, const std::string& label_path, 
             index_t batch_size, bool shuffle)
        : batch_size_(batch_size) {
    read_mnist_images(img_path);
    read_mnist_labels(label_path);
    n_batchs_ = (imgs_.size() + batch_size - 1) / batch_size;

    if(shuffle)
        this->shuffle();
}

void MNIST::get_sample(index_t idx, data_t* sample, index_t& sample_label) const {
    Img* img = reinterpret_cast<Img*>(sample);
    *img = imgs_[idx];
    sample_label = labels_[idx];
}

index_t MNIST::get_batch(index_t idx, data_t* batch, index_t* batch_labels) const {
    index_t n_samples = (idx == n_batchs_ - 1) 
                            ? imgs_.size() - idx * batch_size_
                            : batch_size_;
    auto data = reinterpret_cast<const data_t*>(
        imgs_.data() + idx * batch_size_
    );
    std::memcpy(batch, data, n_samples * sizeof(Img));
    std::memcpy(
        batch_labels, 
        labels_.data() + idx * batch_size_,
        n_samples * sizeof(index_t)
    );
    return n_samples;
}

void MNIST::shuffle(void) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    std::default_random_engine engine1(seed);
    std::shuffle(imgs_.begin(), imgs_.end(), engine1);

    std::default_random_engine engine2(seed);
    std::shuffle(labels_.begin(), labels_.end(), engine2);
}

void MNIST::read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    unsigned int headers[4];
    file.read(reinterpret_cast<char*>(headers), 16);
    index_t magic_number = __reverse_int(headers[0]);
    index_t n_imgs = __reverse_int(headers[1]);
    index_t n_bytes = n_imgs * Img::n_pixels_;

    auto char_data_ptr = std::unique_ptr<char[]>(new char[n_bytes]);
    file.read(char_data_ptr.get(), n_bytes);
    auto uchar_data = reinterpret_cast<unsigned char*>(char_data_ptr.get());

    imgs_.reserve(n_imgs);
    for(index_t i = 0; i < n_imgs; ++i) {
        imgs_.push_back({});
        auto dist = reinterpret_cast<data_t*>(imgs_.data()) + i * Img::n_pixels_;
        auto src = uchar_data + i * Img::n_pixels_;
        for(index_t j = 0; j < Img::n_pixels_; ++j)
            dist[j] = src[j] / 255.0;
    }
}

void MNIST::read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    unsigned int headers[2];
    file.read(reinterpret_cast<char*>(headers), 8);
    index_t magic_number = __reverse_int(headers[0]);
    index_t n_imgs = __reverse_int(headers[1]);
    index_t n_bytes = n_imgs;

    auto char_data_ptr = std::unique_ptr<char[]>(new char[n_bytes]);
    file.read(char_data_ptr.get(), n_bytes);
    auto uchar_data = reinterpret_cast<unsigned char*>(char_data_ptr.get());

    labels_.reserve(n_imgs);
    for(index_t i = 0; i < n_imgs; ++i) {
        index_t label = uchar_data[i];
        labels_.push_back(label);
    }
}
}  // namespace data
}  // namespace st