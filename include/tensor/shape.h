#ifndef TENSOR_SHAPE_H
#define TENSOR_SHAPE_H

#include <initializer_list>
#include <ostream>

#include "utils/base_config.h"
#include "utils/allocator.h"
#include "utils/array.h"

namespace st {

class Shape {
public:
    // constructor
    Shape(std::initializer_list<index_t> dims);
    Shape(const Shape& other, index_t skip);
    Shape(index_t* dims, index_t dim);
    Shape(IndexArray&& shape);
    
    Shape(const Shape& other) = default;
    Shape(Shape&& other) = default;
    ~Shape() = default;

    // method
    index_t dsize() const;
    index_t subsize(index_t start_dim, index_t end_dim) const;
    index_t subsize(index_t start_dim) const;

    // inline function
    index_t ndim(void) const { return dims_.size(); }
    index_t operator[](index_t idx) const { return dims_[idx]; }
    index_t& operator[](index_t idx) { return dims_[idx]; }

    // friend function
    friend std::ostream& operator<<(std::ostream& out, const Shape& s);
private:
    IndexArray dims_;
};

}  // namespace st

#endif