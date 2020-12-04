/*
    Do some simple test here, which is not strict unit test.
*/

#include <iostream>
#include <chrono>

#include "tensor/shape.h"
#include "tensor/storage.h"
#include "tensor/tensor_impl.h"
#include "tensor/tensor.h"
#include "utils/base_config.h"
#include "utils/array.h"
// CHECK_XXX is defined in utils/exception.h
// Only work in namespace st.
#include "utils/exception.h"
#include "exp/operator/function.h"


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

    st::Tensor t1(data, Shape({3, 4}));
    // cout << t1 << endl;
    for(index_t i = 0, idx = -1; i < 3; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            // In fact, TensorImpl::operator[](initilize_list<index_t>) will return 
            // a result of type data_t, but as we know, we can't check whether two 
            // double vars are equal.
            int value = t1[{i, j}];
            CHECK_EQUAL(value, data[++idx], "check 1");
        }
    }
    
    auto t2 = t1.transpose(0, 1);
    // cout << t2 << endl;
    for(index_t i = 0; i < 4; ++i) {
        for(index_t j = 0; j < 3; ++j) {
            int value1 = t1[{j, i}];
            int value2 = t2[{i, j}];
            CHECK_EQUAL(value1, value2, "check 2");
        }
    }

    auto t3 = t2.slice(/*start=*/1, /*end=*/3, /*dim=*/1);
    auto shape_t3 = t3.size();
    CHECK_TRUE(shape_t3[0] == 4 && shape_t3[1] == 2, "check 3");
    for(index_t i = 0; i < 4; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            int value1 = t2[{i, j+1}];
            int value2 = t3[{i, j}];
            CHECK_EQUAL(value1, value2, "check 3");
        }
    }

    auto t4 = t1.view({3, 2, 2});
    auto shape_t4 = t4.size();
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            for(index_t k = 0; k < 2; ++k) {
                int value1 = t1[{i, j*2+k}];
                int value2 = t4[{i, j, k}];
                CHECK_EQUAL(value1, value2, "check 4");
            }
        }
    }

    auto t5 = t4.unsqueeze(/*dim=*/0).unsqueeze(/*dim=*/2);
    CHECK_EQUAL(t5.ndim(), 5, "check 5");
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

    auto t6 = t5.squeeze();
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

void test_operation() {
    using namespace st;

    data_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor t1(data, Shape{3, 4});
    Tensor t2(data, Shape{3, 4});

    Tensor t3 = t1 + t2;
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t3[{i, j}];
            data_t value2 = 2*t1[{i, j}];
            CHECK_EQUAL(value1, value2, "check 1");
        }
    }

    t3 += t1;
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t3[{i, j}];
            data_t value2 = 3*t1[{i, j}];
            CHECK_EQUAL(value1, value2, "check 2");
        }
    }

    Tensor t4 = t1 * t2 + t3;
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t4[{i, j}];
            data_t value2 = t1[{i, j}] * t2[{i, j}] + t3[{i, j}];
            CHECK_EQUAL(value1, value2, "check 3");
        }
    }

    auto func = [&t1, &t2](const Tensor& t3, const Tensor& t4) {
        auto add_exp = t1 + t2;
        auto mul_exp = -t1 * t2;
        return t3 * t4 - add_exp - mul_exp;
    };
    auto exp = func(t3, t4);
    // At this time, add_exp, mul_exp and other implicitly constructed Exp has 
    // been deconstructed. But we expect the BinaryExpImpl hold by them is 
    // still alive, untill the assignment of t5 completes.
    Tensor t5 = exp;
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t5[{i, j}];
            data_t value2 = t3[{i, j}] * t4[{i, j}]
                        - (t1[{i, j}] + t2[{i, j}])
                        - (-t1[{i, j}] * t2[{i, j}]);
            CHECK_EQUAL(value1, value2, "check 3");
        }
    }

    Tensor t6 = t1.view({2, 1, 1, 2, 3});
    Tensor t7 = t1.view({2, 2, 1, 1, 3});
    Tensor t8 = t1.view({2, 2, 3});
    auto exp1 = op::broadcast_add(t6, t7);
    auto exp2 = op::broadcast_mul(t6, -t8);
    auto exp3 = op::broadcast_sub(t6, t8);
    auto exp4 = op::broadcast_add(exp1, exp2);
    auto exp5 = op::broadcast_add(exp4, exp3);
    Tensor t9 = exp5;
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 0; j < 2; ++j)
            for(index_t k = 0; k < 3; ++k)
                for(index_t l = 0; l < 2; ++l) 
                    for(index_t m = 0; m < 3; ++m) {
                        data_t value1 = t9[{i, j, k, l, m}];
                        data_t value2 = t6[{i, 0, 0, l, m}] + t7[{i, j, 0, 0, m}];
                        value2       -= t6[{i, 0, 0, l, m}] * t8[{i, j, k}];
                        value2       += t6[{i, 0, 0, l, m}] - t8[{i, j, k}];
                        CHECK_EQUAL(value1, value2, "check 3");
                    }
}

int main() {
    using namespace st;
    using namespace std::chrono;

    #define BLUE_STRING(s) 

    steady_clock::time_point start_tp = steady_clock::now();

    cout << "\033[33mtest allocator...\33[0m" << endl;
    test_Alloc();

    cout << "\033[33mtest tensor...\33[0m" << endl;
    test_Tensor();

    cout << "\033[33mtest operation...\033[0m" << endl;
    test_operation();

    CHECK_TRUE(Alloc::all_clear(), "check memory all clear");

    steady_clock::time_point end_tp = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(end_tp - start_tp);
    cout << "\033[32mTest success. Test took " << time_span.count();
    cout << " seconds.\033[0m" << endl;
    return 0;
}

