/*
    Do some simple test here, which is not strict unit test.
*/

// The next line can cancel check macro. 
// #define CANCEL_CHECK

#include <iostream>
#include <chrono>

#include "utils/base_config.h"
#include "utils/array.h"
// CHECK_XXX is defined in utils/exception.h
// Only work in namespace st.
#include "utils/exception.h"
#include "tensor/shape.h"
#include "tensor/storage.h"
#include "tensor/tensor_impl.h"
#include "tensor/tensor.h"
#include "exp/function.h"


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
            data_t value = t1[{i, j}];
            CHECK_FLOAT_EQUAL(value, data[++idx], "check 1");
        }
    }
    
    auto t2 = t1.transpose(0, 1);
    // cout << t2 << endl;
    for(index_t i = 0; i < 4; ++i) {
        for(index_t j = 0; j < 3; ++j) {
            data_t value1 = t1[{j, i}];
            data_t value2 = t2[{i, j}];
            CHECK_FLOAT_EQUAL(value1, value2, "check 2");
        }
    }

    auto t3 = t2.slice(/*start=*/1, /*end=*/3, /*dim=*/1);
    auto shape_t3 = t3.size();
    CHECK_TRUE(shape_t3[0] == 4 && shape_t3[1] == 2, "check 3");
    for(index_t i = 0; i < 4; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            data_t value1 = t2[{i, j+1}];
            data_t value2 = t3[{i, j}];
            CHECK_FLOAT_EQUAL(value1, value2, "check 3");
        }
    }

    auto t4 = t1.view({3, 2, 2});
    auto shape_t4 = t4.size();
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            for(index_t k = 0; k < 2; ++k) {
                data_t value1 = t1[{i, j*2+k}];
                data_t value2 = t4[{i, j, k}];
                CHECK_FLOAT_EQUAL(value1, value2, "check 4");
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
                data_t value1 = t4[{i, j, k}];
                data_t value2 = t5[{0, i, 0, j, k}];
                CHECK_FLOAT_EQUAL(value1, value2, "check 5");
            }
        }
    }

    auto t6 = t5.squeeze();
    CHECK_EQUAL(t6.ndim(), t4.ndim(), "check 6");
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            for(index_t k = 0; k < 2; ++k) {
                data_t value1 = t4[{i, j, k}];
                data_t value2 = t6[{i, j, k}];
                CHECK_FLOAT_EQUAL(value1, value2, "check 6");
            }
        }
    }
}

void test_basic_operation() {
    using namespace st;

    data_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor t1(data, Shape{3, 4});
    Tensor t2(data, Shape{3, 4});

    Tensor t3 = t1 + t2;
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t3[{i, j}];
            data_t value2 = 2*t1[{i, j}];
            CHECK_FLOAT_EQUAL(value1, value2, "check 1");
        }
    }

    t3 += t1;
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t3[{i, j}];
            data_t value2 = 3*t1[{i, j}];
            CHECK_FLOAT_EQUAL(value1, value2, "check 2");
        }
    }

    Tensor t4 = t1 * t2 + t3;
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t4[{i, j}];
            data_t value2 = t1[{i, j}] * t2[{i, j}] + t3[{i, j}];
            CHECK_FLOAT_EQUAL(value1, value2, "check 3");
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
            CHECK_FLOAT_EQUAL(value1, value2, "check 3");
        }
    }

    Tensor t6 = t1.view({2, 1, 1, 2, 3});
    Tensor t7 = t1.view({2, 2, 1, 1, 3});
    Tensor t8 = t1.view({2, 2, 3});
    auto exp1 = t6 + t7;
    auto exp2 = t6 * (-t8);
    auto exp3 = t6 - t8;
    Tensor t9 = exp1 + exp2 + exp3;
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 0; j < 2; ++j)
            for(index_t k = 0; k < 3; ++k)
                for(index_t l = 0; l < 2; ++l) 
                    for(index_t m = 0; m < 3; ++m) {
                        data_t value1 = t9[{i, j, k, l, m}];
                        data_t value2 = t6[{i, 0, 0, l, m}] + t7[{i, j, 0, 0, m}];
                        value2    -= t6[{i, 0, 0, l, m}] * t8[{i, j, k}];
                        value2    += t6[{i, 0, 0, l, m}] - t8[{i, j, k}];
                        CHECK_FLOAT_EQUAL(value1, value2, "check 3");
                    }
    
    Tensor t10(Shape{2, 2, 3, 3});
    t10 = t8;
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 0; j < 2; ++j)
            for(index_t k = 0; k < 3; ++k)
                for(index_t l = 0; l < 3; ++l) {
                    data_t value1 = t10[{i, j, k, l}];
                    data_t value2 = t8[{i, j, k}];
                    CHECK_FLOAT_EQUAL(value1, value2, "check 4");
                }
}

void test_matrix_operation() {
    using namespace st;

    data_t data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    data_t data2[] = {11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121};
    Tensor t1(data1, Shape{2, 6});
    Tensor t2(data2, Shape{2, 6});

    Tensor t3 = op::matrix_mul(t1, t2.transpose(/*dim1=*/0, /*dim2=*/1));
    data_t t3_expect[2][2] = {{931, 2191}, {2227, 5647}};
    for(index_t i = 0; i < 2; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            data_t value1 = t3[{i, j}];
            data_t value2 = t3_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check 1");
        }
    }

    Tensor t4 = t1.view({3, 2, 2});
    Tensor t5 = t2.view({3, 2, 2});
    Tensor t6 = op::batch_matrix_mul(t4, t5);
    data_t t6_expect[3][2][2] = {{{73, 103}, {157, 227}}, 
                                {{681, 791}, {925, 1075}},
                                {{1929, 2119}, {2333, 2563}}};
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            for(index_t k = 0; k < 2; ++k) {
                data_t value1 = t6[{i, j, k}];
                data_t value2 = t6_expect[i][j][k];
                CHECK_FLOAT_EQUAL(value1, value2, "check 2");
            }
        }
    }

}

void test_numeric_operation() {
    using namespace st;
    data_t data[] = {0.9742, 0.8367, 0.6840, 1.0074, 1.2784, 1.2193, 
                     1.0252, 1.1873, 1.4498, 1.2189, 0.7510, 1.3621};
    Tensor t0(data, Shape{3, 4});
    data_t log_softmax_expect[3][4] = {{-1.295666, -1.433150, -1.585845, -1.262473},
                                       {-1.289772, -1.348877, -1.542909, -1.380803},
                                       {-1.165368, -1.396256, -1.864143, -1.253019}};
    Tensor t1 = op::log_softmax(t0);
    for(index_t i = 0; i < 3; ++i)
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t1[{i, j}];
            data_t value2 = log_softmax_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check1");
        }
}

int main() {
    using namespace st;
    using namespace std::chrono;

    steady_clock::time_point start_tp = steady_clock::now();

    cout << "\033[33mtest allocator...\33[0m" << endl;
    test_Alloc();

    cout << "\033[33mtest tensor...\33[0m" << endl;
    test_Tensor();

    cout << "\033[33mtest operation...\033[0m" << endl;
    test_basic_operation();

    cout << "\033[33mtest matrix operation...\033[0m" << endl;
    test_matrix_operation();

    cout << "\033[33mtest numeric operation...\033[0m" << endl;
    test_numeric_operation();

    cout << "\033[33mcheck all memory is deallocated...\033[0m" << endl;
    CHECK_TRUE(Alloc::all_clear(), "check memory all clear");

    steady_clock::time_point end_tp = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(end_tp - start_tp);
    cout << "\033[32mTest success. Test took " << time_span.count();
    cout << " seconds.\033[0m" << endl;
    return 0;
}

