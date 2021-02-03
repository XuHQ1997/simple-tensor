### AutoGrad

实现自动求导机制的关键在于动态图的建立，而在实现Expression Template的过程中，我们已经将各个运算抽象出来了，可以视为运算图的节点。接下来只需要处理好节点之间的边就行了。这个边指示了前向传播时的前一个节点，以及反向传播时的下一个节点。在Pytorch中，就是经常看到的`grad_fn`。

以一个简单的例子来说明simple-tensor中反向传播的实现。

```c++
Tensor t0(Shape{2, 3, 4});
Tensor t1(Shape{2, 3, 4});
auto exp0 = t0 + t1;
auto exp1 = exp0 + t1;
Tensor t3 = exp1;
// Pytorch中仅允许从标量张量开始进行反传，
// 下面代码的含义应该是t3.sum().backward()
t3.backward();
```

梯度反传过程如下：

1. `t3.backward`调用后，将初始梯度`d0`设为全1的张量，并调用`t3.grad_fn`。
2. `t3.grad_fn`调用`next_exp_.invoke_backward`，其中`next_exp`是一个`ExpImplPtr`，这里指向`exp1`。
3. `next_exp.invoke_backward`会继续将梯度反传，作为参数传递给`exp1.backward`。
4. `exp1`是一个运算节点，将调用`lhs_ptr_.invoke_backward`和`rhs_ptr_.invoke_backward`。这里的`lhs_ptr_`和`rhs_ptr_`都是`ExpImplPtr`，并且分别指向`exp0`和`t1`。
5. `exp1.lhs_ptr_.invoke_backward`中调用`exp0.backward`。`exp0`与`exp1`相似，最终都是调用其左右运算数的`backward`方法。在`t0.backward`中，会把传入的梯度加到自己保存的梯度上去，`t1.backward`也是一样的。
6. 还有`exp1.rhs_ptr_.invoke_backward`会在调用`t1.backward`，再次累加一次梯度。
7. 至此，从`t3`到`t0`和`t1`的梯度就反传完毕了。

可以看到这个过程有一点深度优先的意思。

##### 处理在计算图中多次出现的张量

在上面的例子中，深度优先没有问题，但是需要考虑下面的情况

```c++
Tensor t0(Shape{2, 3, 4});
Tensor t(Shape{2, 3, 4});
Tensor t1 = t;
auto exp0 = t0 + t1;
auto exp1 = exp0 + t1;
Tensor t3 = exp1;
t3.backward();
```

`t1`在计算图中都出现了两次，不同之处在于这次有了前驱节点`t`。因为`t1`出现了两次，那么`t1.backward`被调用两次是不可避免的，那么每一次`t1.backward`都要进一步调用`t.backward`么？显然是不需要的，并且还可能会导致梯度计算错误。我的实现里是通过`ExpImpl::gradcount_`来进行计数，从而确保只在最后一次调用`backward`的时候，将梯度传递给前驱节点。

##### 计算View所得张量的梯度

```c++
Tensor t0(Shape{2, 3, 4});
Tensor t1 = t0.view({3, 8});
Tensor t2 = t0.transpose(/*dim1=*/0, /*dim2=*/1);
Tensor t3 = t0.slice(0, /*dim=*/1)

Tensor t4 = t1 + t2 + t3;
t4.backward();
```

这个场景中，`t1`到`t3`都是与`t0`共享底层数据的储存空间。那么自然储存梯度的空间也需要共享，并且有一个很好的对应关系：可以用`t1`到`t3`的shape和stride来解释储存梯度的空间。我们知道，一个Tensor可以表示成`(data_ptr, offset, shape, stride)`，改变里面的一个部分就会产生一个不同的Tensor，像`t1`到`t3`就是在`t0`的基础上变换了其中某一部分得到的新的Tensor。前面提到的对应关系是指，`t0`在构造时就开出了自己储存梯度的空间表示为`grad_ptr`，那么`t0`的梯度可以表示为`(grad_ptr, t0.offset, t0.shape, t0.stride)`。那么派生出的张量`t1`可以表示为`(grad_ptr, t1.offset, t1.shape, t1.stride)`，其他派生出的张量也是一样。所以实际上，设计梯度的数据结构时，仅需要保存`grad_ptr`，其他信息Tensor本身已经保存了。

再来看如何从`t4`将梯度反传回`t0`。调用`t4.backward`后，会继续调用`t1`到`t3`的`backward`。那么按照正常的规则，`t1`到`t3`会把传回来的梯度加到自己的梯度上，在把梯度传给前驱节点。但是在这个情境下，因为储存梯度的空间是共享，所以`t1`到`t3`将梯度加到自己的梯度上时，就相当于已经把梯度传给了`t0`，所以`t0`不能把这一份梯度再加一遍。所以要区分前驱节点的`backward`是通过常规的节点调用的，还是通过view、transpose等派生而来的Tensor调用的。

##### Broadcasting机制下的梯度反传

支持Broadcasting机制下的梯度反传相当麻烦。考虑下面的情景：

```c++
Tensor t0(Shape{1, 3, 4});
Tensor t1(Shape{2, 1, 4});
Tensor t2(Shape{2, 3, 1});
Tensor t3 = t0 + t1 + t2;
t3.backward();
```

此时，按照之前说的，用`(t0.grad_ptr, t0.offset, t0.shape, t0.stride)`来表示`t0`的梯度在计算时就已经行不通了。用`grad_t3`来表示`t3`传回来的梯度，其形状是`[2, 3, 4]`，而假设`grad_t0`是`[1, 3, 4]`。我们把`grad_t3`加到`grad_t0`上时，也就是`grad_t0 += grad_t3`，在我的实现中会导致`grad_t0`漏掉一半的梯度。问题的关键在于，`t0`本身不知道`t3`是什么形状，那么自然无法正确地累加自己的梯度，比如通过broadcasting运算所得的`t3`是`[3, 3, 4]`，那么`t0`每个位置的梯度就应该累加3次；如果`t3`是`[4, 3, 4]`，那么就应该累加4次。所以需要将`t3`的形状在反传的时候也要带上。我的实现里是作为额外的函数参数传回去的。