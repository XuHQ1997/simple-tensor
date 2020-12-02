/*
    Do some simple test here, which is not strict unit test.
*/

#include <iostream>

#include "tensor/shape.h"
#include "tensor/storage.h"
#include "tensor/tensor_impl.h"
#include "tensor/tensor.h"
#include "utils/base_config.h"
#include "utils/array.h"
#include "tensor/shape.h"
#include "utils/exception.h"  // CHECK_XXX is defined here.

using std::cout;
using std::endl;

struct Foo {
    static int ctr_call_counter;
    static int dectr_call_counter;

    char x_;
    char y_;
    Foo() { ++ctr_call_counter; }
    Foo(char x, char y) : x_(x), y_(y) { ++ctr_call_counter; }
    ~Foo() { ++dectr_call_counter; }
};
int Foo::ctr_call_counter = 0;
int Foo::dectr_call_counter = 0;

void test_Alloc() {
    using namespace st;
    
    void* ptr;
    {
        // No constructor call here.
        auto uptr = Alloc::unique_allocate<Foo>(sizeof(Foo));
        CHECK_EQUAL(Foo::ctr_call_counter, 0, "check 1");
        ptr = uptr.get();
    }
    CHECK_EQUAL(Foo::dectr_call_counter, 0, "check 1");

    {
        auto sptr = Alloc::shared_allocate<Foo>(sizeof(Foo));
        // The strategy of allocator.
        CHECK_EQUAL(ptr, static_cast<void*>(sptr.get()), "check 2");
    }
    
    {
        auto uptr = Alloc::unique_construct<Foo>();
        CHECK_EQUAL(Foo::ctr_call_counter, 1, "check 3");
        CHECK_EQUAL(ptr, static_cast<void*>(uptr.get()), "check 3");
    }
    CHECK_EQUAL(Foo::dectr_call_counter, 1, "check 3");

    {
        auto sptr = Alloc::shared_construct<Foo>('6', '7');
        CHECK_EQUAL(Foo::ctr_call_counter, 2, "check 4");
        CHECK_TRUE(sptr->x_ == '6' && sptr->y_ == '7', "check 4");
        CHECK_EQUAL(ptr, static_cast<void*>(sptr.get()), "check 4");
    }
    CHECK_EQUAL(Foo::dectr_call_counter, 2, "check 4");
}

void test_Tensor() {
    using namespace st;

    data_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    index_t idata[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    cout << "check 1" << endl;
    st::Tensor t1(data, Shape({3, 4}));
    cout << t1 << endl;
    for(index_t i = 0, idx = -1; i < 3; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            // TensorImpl::operator[](initilize_list<index_t>) will return a result of type data_t,
            // but as we know, we can't check whether two double vars are equal.
            int value = t1[{i, j}];
            CHECK_EQUAL(value, data[++idx], "check 1");
        }
    }
    
    cout << "check 2" << endl;
    auto t2 = t1.transpose(0, 1);
    cout << t2 << endl;
    for(index_t i = 0; i < 4; ++i) {
        for(index_t j = 0; j < 3; ++j) {
            int value1 = t1[{j, i}];
            int value2 = t2[{i, j}];
            CHECK_EQUAL(value1, value2, "check 2");
        }
    }

    cout << "check 3" << endl;
    auto t3 = t2.slice(/*start=*/1, /*end=*/3, /*dim=*/1);
    auto shape_t3 = t3.size();
    cout << shape_t3 << endl;
    CHECK_TRUE(shape_t3[0] == 4 && shape_t3[1] == 2, "check 3");
    cout << t3 << endl;
    for(index_t i = 0; i < 4; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            int value1 = t2[{i, j+1}];
            int value2 = t3[{i, j}];
            CHECK_EQUAL(value1, value2, "check 3");
        }
    }

    cout << "check 4" << endl;
    auto t4 = t1.view({3, 2, 2});
    auto shape_t4 = t4.size();
    cout << shape_t4 << endl;
    cout << t4 << endl;
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            for(index_t k = 0; k < 2; ++k) {
                int value1 = t1[{i, j*2+k}];
                int value2 = t4[{i, j, k}];
                CHECK_EQUAL(value1, value2, "check 4");
            }
        }
    }

    cout << "check 5" << endl;
    auto t5 = t4.unsqueeze(/*dim=*/0).unsqueeze(/*dim=*/2);
    CHECK_EQUAL(t5.ndim(), 5, "check 5");
    cout << t5 << endl;
    Shape shape_t5({1, 3, 1, 2, 2});
    for(index_t i = 0; i < 5; ++i)
        CHECK_EQUAL(t5.size(i), shape_t5[i], "check 5");
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            for(index_t k = 0; k < 2; ++k) {
                int value1 = t4[{i, j, k}];
                int value2 = t5[{0, i, 0, j, k}];
                CHECK_EQUAL(value1, value2, "check 5");
            }
        }
    }

    cout << "check 6" << endl;
    auto t6 = t5.squeeze();
    cout << t6 << endl;
    CHECK_EQUAL(t6.ndim(), t4.ndim(), "check 6");
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            for(index_t k = 0; k < 2; ++k) {
                int value1 = t4[{i, j, k}];
                int value2 = t6[{i, j, k}];
                CHECK_EQUAL(value1, value2, "check 6");
            }
        }
    }
}

int main() {
    using namespace st;
    cout << "test allocator." << endl;
    test_Alloc();

    cout << "test tensor" << endl;
    test_Tensor();

    CHECK_TRUE(Alloc::all_clear(), "check memory all clear");
    cout << "test success" << endl;
    return 0;
}

