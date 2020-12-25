#ifndef DATA_MNIST_H
#define DATA_MNIST_H

#include <string>
#include <vector>

#include "utils/base_config.h"


namespace st {
namespace data {

class DatasetBase {
public:
    virtual index_t n_samples(void) const = 0;
    virtual index_t n_batchs(void) const = 0;
    virtual void get_sample(index_t idx, data_t* sample, index_t& sample_label) const = 0;
    virtual index_t get_batch(index_t idx, data_t* batch, index_t* batch_labels) const = 0;
    virtual void shuffle(void) = 0;
};

class MNIST : public DatasetBase {
public:
    struct Img {
        static constexpr index_t n_rows_ = 28;
        static constexpr index_t n_cols_ = 28;
        static constexpr index_t n_pixels_ = n_rows_ * n_cols_;
        data_t pixels_[n_pixels_];        
    };

    MNIST(const std::string& img_path, const std::string& label_path, 
          index_t batch_size, bool shuffle);
    
    index_t n_samples(void) const override { return imgs_.size(); }
    index_t n_batchs(void) const override { return n_batchs_; }

    void get_sample(index_t idx, data_t* sample, index_t& sample_label) const override;
    index_t get_batch(index_t idx, data_t* batch, index_t* batch_labels) const override;
    void shuffle(void) override;
private:
    void read_mnist_images(const std::string& path);
    void read_mnist_labels(const std::string& path);

    index_t batch_size_, n_batchs_;
    std::vector<Img> imgs_;
    std::vector<index_t> labels_;
};

}  // namespace data
}  // namespace st
#endif