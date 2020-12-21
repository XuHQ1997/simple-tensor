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

    auto t7 = t5.permute({0, 2, 3, 4, 1});
    CHECK_EQUAL(t7.ndim(), 5, "check7");
    Shape shape_t7({1, 1, 2, 2, 3});
    for(index_t i = 0; i < 5; ++i)
        CHECK_EQUAL(t7.size(i), shape_t7[i], "check7");
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 0; j < 2; ++j)
            for(index_t k = 0; k < 3; ++k) {
                data_t value1 = t7[{0, 0, i, j, k}];
                data_t value2 = t5[{0, k, 0, i, j}];
                CHECK_FLOAT_EQUAL(value1, value2, "check7");
            }
}

void test_basic_operator() {
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
    Tensor exp1 = t6 + t7;
    Tensor exp2 = -(t6 * t8);
    Tensor exp3 = t6 - t8;
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

    Tensor t10 = t1.transpose(0, 1) + op::constant(1, {4, 3});
    for(index_t i = 0; i < 4; ++i)
        for(index_t j = 0; j < 3; ++j) {
            data_t value1 = t10[{i, j}];
            data_t value2 = t1[{j, i}] + 1;
            CHECK_EQUAL(value1, value2, "check5");
        }

    // assignment of uncontiguous tensor
    auto t11 = t1.transpose(0, 1);
    Tensor t12 = t2.transpose(0, 1);
    Tensor t13(data, t11.size());
    t11 = t13;
    t12 = t11 + op::constant(0, {4, 3});
    for(index_t i = 0; i < 4; ++i)
        for(index_t j = 0; j < 3; ++j) {
            data_t value1 = t11[{i, j}];
            data_t value2 = t12[{i, j}];
            data_t value3 = t13[{i, j}];
            CHECK_TRUE(value1 == value2 && value1 == value3, "check6");
        }
}

void test_matrix_operator() {
    using namespace st;

    data_t data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    data_t data2[] = {11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121};
    Tensor t1(data1, Shape{2, 6});
    Tensor t2(data2, Shape{2, 6});

    Tensor t3 = op::matrix_transpose(
        op::matrix_mul(t1, t2.transpose(/*dim1=*/0, /*dim2=*/1))
    );
    data_t t3_expect[2][2] = {{931, 2227}, {2191, 5647}};
    for(index_t i = 0; i < 2; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            data_t value1 = t3[{i, j}];
            data_t value2 = t3_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check 1");
        }
    }

    Tensor t4 = t1.view({3, 2, 2});
    Tensor t5 = t2.view({3, 2, 2});
    Tensor t6 = op::batch_matrix_transpose(op::batch_matrix_mul(t4, t5));
    data_t t6_expect[3][2][2] = {{{73, 157}, {103, 227}},
                                 {{681, 925}, {791, 1075}},
                                 {{1929, 2333}, {2119, 2563}}};
    for(index_t i = 0; i < 3; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            for(index_t k = 0; k < 2; ++k) {
                data_t value1 = t6[{i, j, k}];
                data_t value2 = t6_expect[i][j][k];
                CHECK_FLOAT_EQUAL(value1, value2, "check 2");
            }
        }
    }

    Tensor t7 = op::matrix_transpose(t1);
    CHECK_EQUAL(t7.ndim(), 2, "check3");
    CHECK_EQUAL(t7.size(0), 6, "check3");
    CHECK_EQUAL(t7.size(1), 2, "check3");
    for(index_t i = 0; i < 6; ++i) {
        for(index_t j = 0; j < 2; ++j) {
            data_t value1 = t1[{j, i}];
            data_t value2 = t7[{i, j}];
            CHECK_FLOAT_EQUAL(value1, value2, "check 3");
        }
    }

    Tensor t8(data1, Shape{2, 2, 3});
    Tensor t9 = op::batch_matrix_transpose(t8);
    CHECK_EQUAL(t9.ndim(), 3, "check4");
    CHECK_EQUAL(t9.size(0), 2, "check4");
    CHECK_EQUAL(t9.size(1), 3, "check4");
    CHECK_EQUAL(t9.size(2), 2, "check4");
    for(index_t i = 0; i < 2; ++i) {
        for(index_t j = 0; j < 3; ++j) {
            for(index_t k = 0; k < 2; ++k) {
                data_t value1 = t8[{i, k, j}];
                data_t value2 = t9[{i, j, k}];
                CHECK_FLOAT_EQUAL(value1, value2, "check 3");
            }
        }
    }   
}

void test_numeric_operator() {
    using namespace st;
    data_t data1[] = {0.585639, 0.612628, 0.241485, 0.097616, 0.035854, 0.723054, 
                      0.131163, 0.884268, 0.193597, 0.694748, 0.650687, 0.738797};
    Tensor t0(data1, Shape{3, 4});
    data_t t1_expect[3][4] = {{-1.208965, -1.181976, -1.553119, -1.696988},
                              {-1.860054, -1.172853, -1.764744, -1.011639},
                              {-1.784239, -1.283088, -1.327148, -1.239038}};
    Tensor t1 = op::log_softmax(t0);
    for(index_t i = 0; i < 3; ++i)
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t1[{i, j}];
            data_t value2 = t1_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check1");
        }

    auto labels_ptr = Alloc::shared_allocate<index_t>(3 * sizeof(index_t));
    auto labels = labels_ptr.get();
    labels[0] = 2, labels[1] = 0, labels[2] = 3;
    Tensor t2 = op::nll_loss(t1, labels_ptr);
    CHECK_EQUAL(t2.ndim(), 1, "check2");
    CHECK_EQUAL(t2.size(0), t1.size(0), "check2");
    CHECK_FLOAT_EQUAL(t2[{0}], -t1_expect[0][2], "check2");
    CHECK_FLOAT_EQUAL(t2[{1}], -t1_expect[1][0], "check2");
    CHECK_FLOAT_EQUAL(t2[{2}], -t1_expect[2][3], "check2");

    data_t data2[] = {0.096237, -0.037000,  0.028076,  0.328307,  0.122271, -0.017293,
                      0.150791,  0.421008,  0.322066, -0.321352,  0.319534, -0.424081};
    Tensor t3(data2, Shape{2, 2, 3});
    Tensor t4 = op::mean(op::sigmoid(op::relu(t3)), 1);
    data_t t4_expect[][3] = {{0.552694, 0.515265, 0.503509},
                             {0.518813, 0.591467, 0.539914}};
    CHECK_TRUE(t4.ndim() == 2 && t4.size(0) == 2 && t4.size(1) == 3, "check3");
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 0; j < 3; ++j) {
            data_t value1 = t4[{i, j}];
            data_t value2 = t4_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check4");
        }

    Tensor t5 = op::mean(op::mean(t4, 0), 0);
    CHECK_TRUE(t5.ndim() == 1 && t5.size(0) == 1, "check5");
    CHECK_FLOAT_EQUAL(t5.item(), 0.536944, "check5");

    Tensor t6 = op::argmax(t3, 1);
    index_t t6_expect[][3] = {{1, 1, 0},
                              {0, 0, 0}};
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 0; j < 3; ++j) {
            index_t value1 = static_cast<index_t>(t6[{i, j}]);
            index_t value2 = t6_expect[i][j];
            CHECK_EQUAL(value1, value2, "check6");
        }
    
    Tensor t7 = op::max(t3, 1);
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 0; j < 3; ++j) {
            data_t value1 = t7[{i, j}];
            data_t value2 = t3[{i, t6_expect[i][j], j}];
            CHECK_FLOAT_EQUAL(value1, value2, "check7");
        }
}

void test_conv_operator() {
    using namespace st;
    data_t data[6][4] = {{0.4279, 0.7488, 0.3639, 0.5433}, {0.2849, 0.6536, 0.8932, 0.9341}, {0.9640, 0.4822, 0.1887, 0.9457},
                         {0.2132, 0.0185, 0.0163, 0.9874}, {0.2039, 0.8020, 0.3766, 0.6537}, {0.8543, 0.3589, 0.5178, 0.7816}};
    Tensor t0(reinterpret_cast<data_t*>(data), Shape{1, 1, 6, 4});

    Tensor t1 = op::max_pool2d(t0, {2, 2}, {1, 1}, {1, 1});
    index_t t1_size_expect[] = {1, 1, 7, 5};
    data_t t1_expect[7][5] = {{0.4279, 0.7488, 0.7488, 0.5433, 0.5433}, {0.4279, 0.7488, 0.8932, 0.9341, 0.9341},
                              {0.9640, 0.9640, 0.8932, 0.9457, 0.9457}, {0.9640, 0.9640, 0.4822, 0.9874, 0.9874},
                              {0.2132, 0.8020, 0.8020, 0.9874, 0.9874}, {0.8543, 0.8543, 0.8020, 0.7816, 0.7816},
                              {0.8543, 0.8543, 0.5178, 0.7816, 0.7816}};
    CHECK_EQUAL(t1.ndim(), 4, "check1");
    for(index_t i = 0; i < 4; ++i)
        CHECK_EQUAL(t1.size(i), t1_size_expect[i], "check1");
    for(index_t i = 0; i < 7; ++i)
        for(index_t j = 0; j < 5; ++j) {
            data_t value1 = t1[{0, 0, i, j}];
            data_t value2 = t1_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check2");
        }

    Tensor t2 = op::max_pool2d(t1, {3, 4}, {2, 3}, {0, 1});
    index_t t2_size_expect[] = {1, 1, 3, 2};
    data_t t2_expect[][2] = {{0.9640, 0.9457}, {0.9640, 0.9874}, {0.8543, 0.9874}};
    CHECK_EQUAL(t2.ndim(), 4, "check3");
    for(index_t i = 0; i < 4; ++i)
        CHECK_EQUAL(t2.size(i), t2_size_expect[i], "check3");
    for(index_t i = 0; i < 3; ++i)
        for(index_t j = 0; j < 2; ++j) {
            data_t value1 = t2[{0, 0, i, j}];
            data_t value2 = t2_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check3");
        }
    
    Tensor t3 = op::img2col(t0, /*kernel_size=*/{4, 4}, 
                            /*stride=*/{2, 2}, /*padding=*/{1, 1});
    index_t t3_shape_expect[] = {6, 16};
    CHECK_EQUAL(t3.size(0), t3_shape_expect[0], "check4");
    CHECK_EQUAL(t3.size(1), t3_shape_expect[1], "check4");
    data_t t3_expect[][16] = 
        {{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4279, 0.7488, 0.3639, 0.0000, 0.2849, 0.6536, 0.8932, 0.0000, 0.9640, 0.4822, 0.1887},
         {0.0000, 0.0000, 0.0000, 0.0000, 0.7488, 0.3639, 0.5433, 0.0000, 0.6536, 0.8932, 0.9341, 0.0000, 0.4822, 0.1887, 0.9457, 0.0000},
         {0.0000, 0.2849, 0.6536, 0.8932, 0.0000, 0.9640, 0.4822, 0.1887, 0.0000, 0.2132, 0.0185, 0.0163, 0.0000, 0.2039, 0.8020, 0.3766},
         {0.6536, 0.8932, 0.9341, 0.0000, 0.4822, 0.1887, 0.9457, 0.0000, 0.0185, 0.0163, 0.9874, 0.0000, 0.8020, 0.3766, 0.6537, 0.0000},
         {0.0000, 0.2132, 0.0185, 0.0163, 0.0000, 0.2039, 0.8020, 0.3766, 0.0000, 0.8543, 0.3589, 0.5178, 0.0000, 0.0000, 0.0000, 0.0000},
         {0.0185, 0.0163, 0.9874, 0.0000, 0.8020, 0.3766, 0.6537, 0.0000, 0.3589, 0.5178, 0.7816, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000}};
    for(index_t i = 0; i < 6; ++i)
        for(index_t j = 0; j < 16; ++j) {
            data_t value1 = t3[{i, j}];
            data_t value2 = t3_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check4");
        }

    data_t t4_data[2][3][6][4];
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 0; j < 3; ++j)
            for(index_t k = 0; k < 6; ++k)
                for(index_t l = 0; l < 4; ++l)
                    t4_data[i][j][k][l] = data[k][l];
    Tensor t4(reinterpret_cast<data_t*>(t4_data), Shape{2, 3, 6, 4});
    Tensor t5 = op::img2col(t4, /*kernel_size=*/{2, 3}, 
                            /*stride=*/{1, 2}, /*padding=*/{2, 1});
    data_t t5_expect[18][6] = 
        {{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000}, {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000}, {0.0000, 0.0000, 0.0000, 0.0000, 0.4279, 0.7488}, {0.0000, 0.0000, 0.0000, 0.7488, 0.3639, 0.5433}, {0.0000, 0.4279, 0.7488, 0.0000, 0.2849, 0.6536}, {0.7488, 0.3639, 0.5433, 0.6536, 0.8932, 0.9341},
         {0.0000, 0.2849, 0.6536, 0.0000, 0.9640, 0.4822}, {0.6536, 0.8932, 0.9341, 0.4822, 0.1887, 0.9457}, {0.0000, 0.9640, 0.4822, 0.0000, 0.2132, 0.0185}, {0.4822, 0.1887, 0.9457, 0.0185, 0.0163, 0.9874}, {0.0000, 0.2132, 0.0185, 0.0000, 0.2039, 0.8020}, {0.0185, 0.0163, 0.9874, 0.8020, 0.3766, 0.6537},
         {0.0000, 0.2039, 0.8020, 0.0000, 0.8543, 0.3589}, {0.8020, 0.3766, 0.6537, 0.3589, 0.5178, 0.7816}, {0.0000, 0.8543, 0.3589, 0.0000, 0.0000, 0.0000}, {0.3589, 0.5178, 0.7816, 0.0000, 0.0000, 0.0000}, {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000}, {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000}};
    CHECK_EQUAL(t5.size(0), 36, "check5");
    CHECK_EQUAL(t5.size(1), 18, "check5");
    for(index_t i = 0; i < 36; ++i)
        for(index_t j = 0; j < 18; ++j) {
            data_t value1 = t5[{i, j}];
            data_t value2 = t5_expect[i/2][j%6];
            CHECK_FLOAT_EQUAL(value1, value2, "check5");
        }
}

void test_tensor_backward() {
    using namespace st;

    data_t data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor t0(data1, Shape{3, 4}, true);

    Tensor t1 = t0.view({2, 2, 3});
    Tensor t2 = t1.slice(/*start_idx=*/1, /*end_idx=*/3, /*dim=*/2);
    Tensor t3 = t2.slice(1, /*dim=*/1);
    t3.backward();
    data_t t0_grad_expect1[][4] = {{0, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 1, 1}};
    auto&& grad1 = t0.grad();
    for(index_t i = 0; i < 3; ++i)
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = grad1[{i, j}];
            data_t value2 = t0_grad_expect1[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check1");
        }

    Tensor t4 = t0.view({3, 2, 2});
    Tensor t5 = t4.transpose(0, 1);
    Tensor t6 = t5.slice(0, 1, /*dim=*/0);
    Tensor t7 = t6.permute({1, 2, 0});
    t7.backward();
    data_t t0_grad_expect2[][4] = {{1, 1, 0, 0}, {2, 2, 0, 0}, {1, 1, 1, 1}};
    auto&& grad2 = t0.grad();
    for(index_t i = 0; i < 3; ++i)
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = grad2[{i, j}];
            data_t value2 = t0_grad_expect2[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check2");
        }
}

void test_basic_operator_backward() {
    using namespace st;

    data_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor t0(data, Shape{3, 4}, /*requires_grad=*/true);
    Tensor t1(data, Shape{3, 4}, /*requires_grad=*/true);

    Tensor t2 = t0 + t1;
    Tensor t3 = t0 * (-t2);
    Tensor t4 = t3 - t2;
    t4.backward();

    data_t t0_grad_expect[][4] = {{-4, -7, -10, -13}, {-16, -19, -22, -25}, {-28, -31, -34, -37}};
    data_t t1_grad_expect[][4] = {{-2, -3, -4, -5}, {-6, -7, -8, -9}, {-10, -11, -12, -13}};
    auto&& t0_grad = t0.grad();
    auto&& t1_grad = t1.grad();
    for(index_t i = 0; i < 3; ++i)
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t0_grad[{i, j}];
            data_t value2 = t1_grad[{i, j}];
            data_t value3 = t0_grad_expect[i][j];
            data_t value4 = t1_grad_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value3, "check1");
            CHECK_FLOAT_EQUAL(value2, value4, "check1");
        }
}

void test_matrix_operator_backward() {
    using namespace st;

    data_t data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    data_t data2[] = {11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121};
    Tensor t1(data1, Shape{2, 6}, true);
    Tensor t2(data2, Shape{2, 6}, true);

    Tensor t3 = op::matrix_mul(t1, t2.transpose(/*dim1=*/0, /*dim2=*/1));
    t3.backward();
    
    Tensor t4 = t1.view({3, 2, 2});
    Tensor t5 = t2.view({3, 2, 2});
    Tensor t6 = op::batch_matrix_mul(t4, t5);
    t6.backward();
    
    data_t t1_grad_expect[][6] = {{114., 174., 154., 214., 274., 334.}, {194., 254., 314., 374., 354., 414.}};
    data_t t2_grad_expect[][6] = {{12., 14., 18., 20., 28., 30.}, {22., 24., 32., 34., 38., 40.}};
    auto&& t1_grad = t1.grad();
    auto&& t2_grad = t2.grad();
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 6; j < 6; ++j) {
            data_t value1 = t1_grad[{i, j}];
            data_t value2 = t2_grad[{i, j}];
            CHECK_FLOAT_EQUAL(value1, value2, "check1");
        }
}

void test_numeric_operator_backward() {
    using namespace st;

    data_t data1[] = {0.585639, 0.612628, 0.241485, 0.097616, 0.035854, 0.723054, 
                      0.131163, 0.884268, 0.193597, 0.694748, 0.650687, 0.738797};
    Tensor t0(data1, Shape{3, 4}, /*requires_grad=*/true);
    Tensor t1 = op::log_softmax(t0);

    auto labels_ptr = Alloc::shared_allocate<index_t>(3 * sizeof(index_t));
    auto labels = labels_ptr.get();
    labels[0] = 2, labels[1] = 0, labels[2] = 3;
    Tensor t2 = op::nll_loss(t1, labels_ptr);

    t2.backward();
    data_t t0_grad_expect[][4] = {{0.2985, 0.3067, -0.7884, 0.1832},
                                  {-0.8443, 0.3095, 0.1712, 0.3636},
                                  {0.1679, 0.2772, 0.2652, -0.7103}};
    auto&& t0_grad = t0.grad();
    for(index_t i = 0; i < 3; ++i) 
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t0_grad[{i, j}];
            data_t value2 = t0_grad_expect[i][j];
            CHECK_FLOAT_EQUAL(value1, value2, "check1");
        }

    data_t data2[] = {0.096237, -0.037000,  0.028076,  0.328307,  0.122271, -0.017293,
                      0.150791,  0.421008,  0.322066, -0.321352,  0.319534, -0.424081};
    Tensor t3(data2, Shape{2, 2, 3}, /*requires_grad=*/true);
    Tensor t4 = op::mean(op::sigmoid(op::relu(t3)), 1);
    Tensor t5 = op::max(t4, 1);

    t5.backward();
    data_t t3_grad_expect[][2][3] = {{{0.1247, 0.0000, 0.0000}, {0.1217, 0.0000, 0.0000}},
                                     {{0.0000, 0.1196, 0.0000}, {0.0000, 0.1219, 0.0000}}};
    auto&& t3_grad = t3.grad();
    for(index_t i = 0; i < 2; ++i)
        for(index_t j = 0; j < 2; ++j)
            for(index_t k = 0; k < 3; ++k) {
                data_t value1 = t3_grad[{i, j ,k}];
                data_t value2 = t3_grad_expect[i][j][k];
                CHECK_FLOAT_EQUAL(value1, value2, "check2");
            }
}

void test_img2col_operator_backward() {
    using namespace st;
    data_t data[6][4] = {{0.4279, 0.7488, 0.3639, 0.5433}, {0.2849, 0.6536, 0.8932, 0.9341}, {0.9640, 0.4822, 0.1887, 0.9457},
                         {0.2132, 0.0185, 0.0163, 0.9874}, {0.2039, 0.8020, 0.3766, 0.6537}, {0.8543, 0.3589, 0.5178, 0.7816}};
    Tensor t0(reinterpret_cast<data_t*>(data), Shape{1, 1, 6, 4}, true);

    Tensor t1 = op::img2col(t0, /*kernel_size=*/{5, 3},
                            /*stride=*/{1, 1}, /*padding=*/{0, 0});
    t1.backward();

    Tensor t2 = op::img2col(t0, /*kernel_size=*/{3, 3},
                            /*stride=*/{1, 1}, /*padding=*/{1, 1});
    t2.backward();

    auto&& t0_grad = t0.grad();
    data_t t0_grad_expect[][4] = {{5., 8., 8., 5.}, {8., 13., 13., 8.}, {8., 13., 13., 8.},
                                  {8., 13., 13., 8.}, {8., 13., 13., 8.}, {5., 8., 8., 5.}};
    for(index_t i = 0; i < 6; ++i) {
        for(index_t j = 0; j < 4; ++j) {
            data_t value1 = t0_grad[{0, 0, i, j}];
            data_t value2 = t0_grad_expect[i][j];
            CHECK_EQUAL(value1, value2, "check1");
        }
    }
}

int main() {
    using namespace std::chrono;

    steady_clock::time_point start_tp = steady_clock::now();

    cout << "\033[33mtest allocator...\33[0m" << endl;
    test_Alloc();

    cout << "\033[33mtest tensor...\33[0m" << endl;
    test_Tensor();

    cout << "\033[33mtest basic operator...\033[0m" << endl;
    test_basic_operator();

    cout << "\033[33mtest matrix operator...\033[0m" << endl;
    test_matrix_operator();

    cout << "\033[33mtest numeric operator...\033[0m" << endl;
    test_numeric_operator();

    cout << "\033[33mtest conv operator...\033[0m" << endl;
    test_conv_operator();

    cout << "\033[33mtest tensor backward...\033[0m" << endl;
    test_tensor_backward();

    cout << "\033[33mtest basic operator backward...\033[0m" << endl;
    test_basic_operator_backward();

    cout << "\033[33mtest matrix operator backward...\033[0m" << endl;
    test_matrix_operator_backward();

    cout << "\033[33mtest numeric operator backward...\033[0m" << endl;
    test_numeric_operator_backward();

    cout << "\033[33mtest img2col operator backward...\033[0m" << endl;
    test_img2col_operator_backward();

    cout << "\033[33mcheck all memory is deallocated...\033[0m" << endl;
    CHECK_TRUE(st::Alloc::all_clear(), "check memory all clear");

    steady_clock::time_point end_tp = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(end_tp - start_tp);
    cout << "\033[32mTest success. Test took " << time_span.count();
    cout << " seconds.\033[0m" << endl;
    return 0;
}
