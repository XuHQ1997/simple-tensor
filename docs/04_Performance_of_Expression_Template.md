本项目编码上最复杂的点就是Expression Template，也就是利用模板实现静态多态。这里与常规的虚函数实现多态做个比较，看看性能会好多少。因为用虚函数重新实现一遍项目的话工作量太大，所以只能做一个简单的demo进行实验，对优化效果有一个大致的预期。

### 测试说明

这个demo只考虑3个的一维向量，只考虑加减乘三种运算操作。测试流程是，进行10次循环，每次循环里对向量进行一次初始化，再做1000次运算，每次运算都是从给定的表达式中随机选择一个执行的，简略地表示如下。

```c++
Vec op1, op2, op3;
vector<std::function<void(void)>> expressions{
    [&]{ op1 = op1 + op2 + op3; }
    ...
    [&]{ op1 = op2; }
};

for(int i = 0; i < 10; ++i) {
	init(op1), init(op2), init(op3);
    for(int j = 0; j < 1000; ++j) {
        int choice = random_choice();
        expressions[choice]();
    }
}
```

如文档03_Template_Expression所述，为了实现惰性计算和避免额外的空间开销，函数调用路径稍微会复杂一些。以`op1=op1+op2+op3`这个表达式为例，`op1+op2`会返回一个`class AddExp`的对象`temp1`，然后`temp1+op3`会再返回一个`class AddExp`的对象`temp2`。所有实际的计算发生在`op1=temp2`这个函数中。

```c++
Vec* Vec::operator=(const Exp& exp) {
	for(int i = 0; i < dim; ++i)
		(*this)[i] = exp.eval(i);
	return *this;
}
```

在demo中，`op1`含有1000个元素，所以`temp2.eval()`会被调用1000次。`temp2.eval()`又会继续调用它的两个操作数`temp1`和`op3`的`eval`方法，`temp1.eval()`又会继续调用它的两个操作数`op1`和`op2`的`eval`方法。可以看到每次赋值操作都会调用多次`eval()`，所以其调用效率会极大地影响程序性能。

### 基于虚函数的多态实现

```c++
class Exp {
public:
	virtual data_t eval(int idx) const = 0;
};
class BinaryExp : public Exp {
    virtual data_t eval(int idx) const override;
};
class Vec : public Exp {
    virtual data_t eval(int idx) const override;
}
```

实现`class Exp`作为基类，然后`class BinaryExp`和`class Vec`作为派生类，并实现各自的`eval`作为虚函数。上面所述的`Vec& Vec::operator=(const Exp&)`会通过虚函数的方式调用派生类相应的`eval`方法。具体代码见`./dynamic_expression_test.cpp`。

### Template Expression的实现

```c++
template<typename SubType>
class Exp {
public:
    data_t eval(int idx) const { return self()->eval(idx); }
private:
    const SubType* self(void) const { 
        return static_cast<const SubType*>(this); 
    }
};

template<typename LhsType, typename RhsType>
class BinaryExp {
public:
    data_t eval(int idx) const;
};

class Vec {
public:
    data_t eval(int idx) const;
    template<typename Subtype> Vec& operator=(const Exp<Subtype>&);
};
```

因为基类使用了模板，很多地方都得跟着变成模板，这也是这个实现的缺点。`operator=`调用`eval`时，因为直接通过模板参数拿到了`Subtype`信息，所以不需要使用虚函数，就可以直接调用派生类的`eval`方法。具体代码见`./static_expression_test.cpp`。

### 运行效率比较

| 实现               | O0编译运行时间 | O0编译后文件大小 | O2编译运行时间 | O2编译后文件大小 |
| ------------------ | -------------- | ---------------- | -------------- | ---------------- |
| 虚函数             | 82.16ms        | 72376B           | 31.37ms        | 30584B           |
| TemplateExpression | 139.58ms       | 73912B           | 4.09ms         | 25528B           |

可以看到在O2优化下，TemplateExpression比虚函数快了将近8倍，说明这项技术在本项目中确实能够发挥很大的优化作用。

为了便于分析效率提升的来源，我们重写main函数，只一次表达式`op1 = op2 + op3 - op1`。通过观察O2优化下的汇编代码可以看到，首先`Vec::operator=()`这个函数都被内联到了main函数体中。而最明显的不同是，虚函数的实现在循环中有两个call指令，并且在call指向函数体中，还有call+寄存器的指令，这显然是在查找虚函数；而template expression的实现在循环中全部都是算数指令，没有任何的函数调用。所以template expression的实现不需要保存任何函数体，这也解释了为什么O2优化下可执行文件体积上的差异。

### 实际项目优化效果的推算

在上面的说明中，我们可以看到，template expression对于项目最大的优化在于eval函数的调用。在我们的demo中，`eval`调用了34114000次，那么平均下来每一千万次调用可以快7.99ms。

在实际项目中，以训练一个周期的MLP为例，`eval`会被调用327851108次，耗时585909ms。那么在eval调用上可以节约262.17ms，也就是template expression可以带来4.5%左右的效率提升。虽然还有其他地方也会被优化，但也不会差太多了。不得不说这个数值比想象中还是差很多QAQ。