#ifndef DATA_MNIST_H
#define DATA_MNIST_H

#include <string>
#include <vector>
#include <tuple>

#include "utils/base_config.h"


namespace st {
namespace data {

class DatasetBase {
public:
    virtual index_t n_samples(void) const = 0;
    virtual index_t n_batchs(void) const = 0;
    
    virtual std::pair<const data_t*, index_t> get_sample(index_t idx) const = 0;
    virtual std::tuple<index_t, const data_t*, const index_t*> 
    get_batch(index_t idx) const = 0;
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

    std::pair<const data_t*, index_t> get_sample(index_t idx) const override;
    std::tuple<index_t, const data_t*, const index_t*> 
    get_batch(index_t idx) const override;
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