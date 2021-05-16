### Expression Template

Expression Template是奇异递归模板模式（Curiously Recurring Template Pattern, CRTP）的一个应用，是C++模板编程的一种技巧。强烈推荐这个[tutorial](https://github.com/dmlc/mshadow/tree/master/guide/exp-template)，十分简单易懂。下面会换个与它不同的思路介绍。

考虑这样一个情况：

```python
# Python code
>>> t0, t1, t2 = [torch.rand(2, 3, 4) for _ in range(3)]
>>> t3 = t0 + t1 + t2
```

`t0 + t1 + t2`这个表达式应该如何处理是我们关注的点。

##### 1. Naive Bad Solution

如果是1+2+3这样3个整数加起来的表达式，我们可以设计成1+2得到一个整数3，然后3+3得到最终结果6。但是3个Tensor加起来就不一样了。如果`t0+t1`也返回一个Tensor，我们需要一个临时的空间储存`t0+t1`所得到的每个元素的结果，然后拿这个临时空间里的数据和`t2`再进行运算。用简化的c++代码将这个过程可以表示如下：

```c++
Tensor t0(2, 3, 4);
Tensor t1(2, 3, 4);
Tensor t2(2, 3, 4);

Tensor temp(2, 3, 4);
for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 3; ++j)
        for(int k = 0; k < 4; ++k)
            temp[i, j, k] = t0[i, j, k] + t1[i, j, k];

Tensor t3(2, 3, 4);
for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 3; ++j)
        for(int k = 0; k < 4; ++k)
            t3[i, j, k] = temp[i, j, k] + t2[i, j, k];
```

这显然不是一个好的解决方案，一方面是需要额外的临时空间（在一些情况下，可以用移动语义避免额外临时空间，但仍不理想，比如当t3是已经构造出来的Tensor时），一方面是用了两个循环才做完。明确一下我们的目标，我们期待`t0+t1+t2`这个表达式的底层执行过程应该是：

```c++
for(int i = 0; i < 2; ++i)
    for(int j = 0; j < 3; ++j)
        for(int k = 0; k < 4; ++k)
            t3[i, j, k] = t0[i, j, k] + t1[i, j, k] + t2[i, j, k];
```

##### 2. Dynamic Polymorphism（动态多态）

那么如何实现呢？关键在于，看到`t0+t1`的时候不进行实际运算，看完整个表达式后，再进行计算。我们可以让`t0+t1`返回一个中间对象，这里用`class AddExp`的实例`exp1`表示，这个过程不进行任何计算，仅仅记录下参与运算的Tensor是谁，然后处理`exp1 + t2`，然后继续返回一个AddExp的实例`exp2`。然后我们要处理的就是`t3=exp2`，在这个过程中进行实际运算，并直接将结果存入`t3`的储存空间中。有了这个思路后，剩下的就是`AddExp`如何定义，以及操作符重载的问题了。

因为AddExp中需要记录两个操作对象是谁，可能是Tensor，也可能是另一个AddExp，所以Tensor和AddExp之间必须实现多态。下面的代码做了简化，只考虑了形状是`(2, 3, 4)`的Tensor。

```c++
// 抽象类，作为多态中的基类
struct Exp {
    virtual double eval(int i, int j, int k) const = 0;
    virtual ~Exp() = default;
};

struct Tensor : public Exp {
    // 省略了其他成员函数
    ...
    // 重写基类的纯虚函数
    // 表示取自己[i,j,k]的元素
    double eval(int i, int j, int k) const override {
        return (*this)[i, j, k];
    }
    // 重载自己的赋值运算符，此时进行真正的运算
    // 把各个元素值算出来，并存起来
    Tensor operator=(const Exp& exp) {
        for(int i = 0; i < 2; ++i) 
            for(int j = 0; j < 3; ++j)
                for(int k = 0; k < 4; ++k)
                    (*this)[i, j, k] = exp.eval(i, j, k);
    }
};

struct AddExp : public Exp {
    // 通过指针保存操作数
    Exp* loperand, roperand;
    // 构造函数，保存操作数
    AddExp(Exp* l, Exp* r) : loperand(l), roperand(r) {}
    // 重写基类的纯虚函数
    double eval(int i, int j, int k) const override {
        return loperand->eval(i, j, k) + roperand->eval(i, j, k);
    }
};
```

然后是操作符重载的问题，要重载加号运算符，让`t0+t1`和`exp1+t2`都返回一个`AddExp`。

```c++
AddExp operator+(const Exp& e1, const Exp& e2) {
    return AddExp(&e1, &e2);
}
```

这样的话，再回来看`t3 = t0 + t1 + t2 `这个表达式：

1. `t0 + t1`，根据操作符重载，调用`operator+(t0, t1)`，返回AddExp的临时对象`exp1`
2. `exp1 + t2`，根据操作符重载，调用`operator(exp1, t2)`，返回AddExp的临时对象`exp2`
3. `t3 = exp2`，调用`Tensor::operator=(exp2)`。其中，会调用`exp2.eval(i, j, k)`，而
   - `exp2.eval(i,j,k) = exp1.eval(i,j,k) + t2.eval(i,j,k)`
   - `exp2.eval(i,j,k) = t1.eval(i,j,k) + t2.eval(i,j,k) + t3.eval(i,j,k)`
   - `exp2.eval(i,j,k) = t1[i,j,k] + t2[i,j,k] + t3[i,j,k]`

这个过程看起来很理想，尤其是假设编译器会做函数内联优化的时候。但是其中引入了虚函数，函数内联得不到保证，而且还要承担查找虚表等开销，效率不见得会比我们最开始提到的实现好。这时，奇异递归模板模式（CRTP）带来了福音，我们可以通过CRTP实现静态多态，把多态的事情放在编译期，交给编译器去做。

##### 3. Expression Template

CRTP的核心作于，将派生类作为基类模板的模板参数，从而使基类模板记录派生类的信息。之前通过虚函数实现的动态多态，是通过派生类的虚表记录运行时的多态信息；而CRTP实现的静态多态，是通过基类的模板参数在编译期记录派生类的信息。在我们的场景中，可以实现如下：

```c++
template<typename Subtype>
struct Exp {
    // 这个self()函数是CRTP的精髓所在
    Subtype* self(void) {
        return static_cast<Subtype*>(this);
    };
    
    double eval(int i, int j, int k) const {
        // 将自己转成派生类，就可以调用派生类的eval函数了
        return self()->eval(i, j, k);
    }
};

struct Tensor : public Exp<Tensor> {
    // 省略了其他成员函数
    ...
    // 实现eval()，需要保证接口和Exp里面的调用一致
    double eval(int i, int j, int k) const {
        return (*this)[i, j, k];
    }
    
    // 因为Exp现在变成模板了，这里也要变成模板函数
    template<typename SubType>
    Tensor operator=(const Exp<SubType>& exp) {
        for(int i = 0; i < 2; ++i) 
            for(int j = 0; j < 3; ++j)
                for(int k = 0; k < 4; ++k)
                    (*this)[i, j, k] = exp.eval(i, j, k);
    }
};

template<typename LhsType, typename RhsType>
struct AddExp : public Exp<AddExp<LhsType, RhsType>> {
    // 通过指针保存操作数
    Exp<LhsType>* loperand;
    Exp<RhsType>* roperand;
    // 构造函数，保存操作数
    AddExp(Exp<LhsType>* l, Exp<RhsType>* r) 
            : loperand(l), roperand(r) {}
    // 实现eval方法，必须保证接口一致
    double eval(int i, int j, int k) const {
        return loperand->eval(i, j, k) + roperand->eval(i, j, k);
    }
};
```

现在的实现没有任何虚函数，所有`eval()`的调用都在编译期被编译成了我们需要的派生类的`eval`方法，函数内联可以得到了可靠的保证，基本达到了我们的期待。（代价就是模板编程太麻烦了。。）

##### 4. Some issues

通过上面的简单场景，我们介绍了基本概念。而在稍微复杂场景下，实现上还可能存在问题。比如：

```c++
AddExp exp = t0 + t1 + t2;
t3 = exp;
```

这里仅仅是把`t3 = t0 + t1 + t2`断开了，就可能存在风险（也许实际上并不会有问题，但肯定还是得避免）。因为`t0+t1`产生的临时对象，可能在`t3 = exp`执行之前被析构掉了。`t3 = t0 + t1 + t2`是一行代码，我们可以确信其中的临时对象绝对不会中途析构，但现在断开以后就无法确定这件事了（是的，这就是c++）。

还有一个更加明显的场景：

```c++
// 这里用了lambda表达式，换成正常的函数也是一个意思
auto func = [](Tensor& t0, Tensor& t1, Tensor& t2) {
    return t0 + t1 + t2;
};

t3 = func(t0, t1, t2);
```

这里`t0 + t1`产生的临时对象，无需置疑，一定会在`t3.operator=`被调用前被析构掉。实际运行起来可能不会出问题，因为AddExp的析构函数可能是空的，但是我们肯定是得避免这个问题。

要解决这个问题，就不能让`AddExp`这个类的对象生存在栈上，需要让它们构造在堆内存中。比如将加号运算符重载简单地改写成：

```c++
AddExp* operator+(const Exp& e1, const Exp& e2) {
    return new AddExp(&e1, &e2);
}
```

这样带来的问题是，`t0 + t1 + t2`这个表达式无法通过编译，因为`t0 + t1`返回的是指针，我们并没有对指针做运算符重载。这时就得用Pimpl设计模式了（其实Pimpl模式并不是解决这个问题的，我是解决这个问题之后才看到的这个名词）。也就是用另一个类`class AddExpShell`来维护`class AddExp`的指针，这样`AddExpShell`生存在栈空间里，`AddExp`生存在堆空间里。同理，`TensorShell`乃至`ExpShell`都是需要的。而把对象构造在堆上又带来了何时析构的问题，又不得不使用智能指针或者引用计数。

##### 5. My Implement

在Simple-Tensor中，相关代码主要位于`include/exp`目录下。`TensorImpl`是实际的数据，生存在堆空间中；`Tensor`是前面提到的`TensorShell`。各种`ExpImpl`也是一样。

然后，为了便于拓展把`AddExp`这个类分解成了两个层次：

```c++
template<typename Op, typename LhsType, typename RhsType>
class BinaryExpImpl: public ExpImpl<BinaryExpImpl<Op, LhsType, RhsType>> {
    // 省略一部分代码
    ...
    // 实现eval函数，将具体计算委托给Op的静态函数，把操作数和索引传过去
    // 这里的IndexArray& inds就相当与前面一直用的[i, j, k]
    double eval(IndexArray& inds) const {
        return Op<LhsType, RhsType>::eval(inds, lhs, rhs);
    }
};

template<typename LhsType, typename RhsType>
class Add {
    static double eval(IndexArray& inds, LhsType& lhs, RhsType& rhs) {
        return lhs.eval(inds) + rhs.eval(inds);
    }
};
```

因为现在数据构造在堆上，而且感觉`std::shared_ptr`不太好用，所以自己实现了`ExpImplPtr`来维护引用计数，控制何时析构对象，而且也在后面实现自动求导的时候发挥了作用。
