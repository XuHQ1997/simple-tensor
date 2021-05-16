#include <iostream>
#include <chrono>
#include <thread>
#include <random>
#include <vector>
#include <ctime>
#include <functional>
using namespace std;

using data_t = double;

class Exp {
public:
    virtual data_t eval(int idx) const = 0;
};

class Vec : public Exp {
public:
    static const int dim = 1000;

    virtual data_t eval(int idx) const override { return nums[idx]; }
    data_t& operator[](int idx) { return nums[idx]; }

    Vec& operator=(const Exp& exp) {
        for(int i = 0; i < dim; ++i)
            nums[i] = exp.eval(i);
        return *this;
    }
private:
    data_t nums[dim];
};

template<typename Op>
class BinaryExp : public Exp {
public:
    BinaryExp(const Exp& lhs, const Exp& rhs)
            : rhs(rhs), lhs(lhs) {}
    virtual data_t eval(int idx) const override {
        return Op::eval(lhs.eval(idx), rhs.eval(idx));
    }
private:
    const Exp& lhs;
    const Exp& rhs;
};

struct Add {
    static data_t eval(data_t a, data_t b) { return a + b; }
};
struct Sub {
    static data_t eval(data_t a, data_t b) { return a - b; }
};
struct Mul {
    static data_t eval(data_t a, data_t b) { return a * b; }
};

BinaryExp<Add> operator+(const Exp& lhs, const Exp& rhs) {
    return BinaryExp<Add>(lhs, rhs);
}
BinaryExp<Sub> operator-(const Exp& lhs, const Exp& rhs) {
    return BinaryExp<Sub>(lhs, rhs);
}
BinaryExp<Mul> operator*(const Exp& lhs, const Exp& rhs) {
    return BinaryExp<Mul>(lhs, rhs);
}


int main() {
    constexpr int n_loops = 10;
    constexpr int n_operations = 1000;

    Vec op1, op2, op3;
    vector<std::function<void(void)>> expressions{
        [&]{ op1 = op1 + op2 + op3; },
        [&]{ op2 = op1 - op2 - op3; },
        [&]{ op3 = op1 * op2 + op3; },
        [&]{ op1 = op1 - op2 - op3; },
        [&]{ op2 = op1 + op2 + op3; },
        [&]{ op1 = op2 + op1; },
        [&]{ op2 = op1 * op2; },
        [&]{ op3 = op1; },
        [&]{ op1 = op2; }
    };

    default_random_engine e(time(nullptr));
    uniform_real_distribution<double> number_u(-1, 1);
    uniform_int_distribution<int> choice_u(0, expressions.size()-1);

    auto start_tp = chrono::steady_clock::now();
    for(int i = 0; i < n_loops; ++i) {
        for(int j = 0; j < Vec::dim; ++j) {
            op1[j] = number_u(e);
            op2[j] = number_u(e);
            op3[j] = number_u(e);
        }

        for(int j = 0; j < n_operations; ++j) {
            int choice = choice_u(e);
            expressions[choice]();
        }
    }
    auto end_tp = chrono::steady_clock::now();
    chrono::duration<double, std::milli> d = end_tp - start_tp;
    cout << d.count() << endl;
    return 0;
}