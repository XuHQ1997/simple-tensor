#include "tensor/shape.h"

namespace st {

Shape::Shape(std::initializer_list<index_t> dims) : dims_(dims) {}

Shape::Shape(const Shape& other, index_t skip) : dims_(other.ndim() - 1) {
    int i = 0;
    for(; i < skip; ++i)
        dims_[i] = other.dims_[i];
    for(; i < dims_.size(); ++i)
        dims_[i] = other.dims_[i+1];
}

Shape::Shape(index_t* dims, index_t dim_) : dims_(dims, dim_) {}

Shape::Shape(IndexArray&& shape) : dims_(std::move(shape)) {}

index_t Shape::dsize() const {
    int res = 1;
    for(int i = 0; i < dims_.size(); ++i)
        res *= dims_[i];
    return res;
}

index_t Shape::subsize(index_t start_dim, index_t end_dim) const {
    int res = 1;
    for(; start_dim < end_dim; ++start_dim)
        res *= dims_[start_dim];
    return res;
}

index_t Shape::subsize(index_t start_dim) const {
    return subsize(start_dim, dims_.size());
}

bool Shape::operator==(const Shape& other) const {
    if(this->ndim() != other.ndim()) return false;
    index_t i = 0;
    for(; i < dims_.size() && dims_[i] == other.dims_[i]; ++i)
        ;
    return i == dims_.size();
}

std::ostream& operator<<(std::ostream& out, const Shape& s) {
    out << '(' << s[0];
    for(int i = 1; i < s.ndim(); ++i)
        out << ", " << s[i];
    out << ")";
    return out;
}

}  // namespace st