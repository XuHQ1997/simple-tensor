# Simple-Tensor

Simple-Tensor is trying to implement **Basic operation of Tensor** ,**Autograd based on Dynamic Computational Graph** and **simple Neural Network** in c++.

For many reasons, the repositories such as Pytorch and Tensorflow is too complicated to learn about how Tensors and Computational graphs work. While in this repository, I try to make the concept complete and the code clear.  And of course you shouldn't expect this repo is as efficient as Pytorch or others.

### Features

- [x] Memory management
- [x] Basic operations of Tensor
- [x] Expression Templates
- [x] Autograd  based on Dynamic Computational Graph
- [x] Simple Neural Network
- [x] Initializer and Optimizer of NN

### Experiment

##### 1. MLP with three linear layers trained on MNSIT

|  Initializer   |    Optimizer    | Epoch |  Time   |  ACC   |
| :------------: | :-------------: | :---: | :-----: | :----: |
| KaimingUniform | SGDwithMomentum |   2   | 1125.9s | 0.9708 |

The training process wasn't fine-tuned. So I think there is still some room for improvement.

##### 2. ConvNet trained on MNIST

TODO

### Build and Run

I didn't use any special feature of compiler, so I think it's easy to build. And my development environment is as follows:

```
OS: Windows 10
Compiler: g++ from MinGW-W64-builds-4.3.5
Make: mingw32-make from MinGW-W64-builds-4.3.5
```

##### 1. Unit Test

To build the project and run test:

``` shell
# 1. clone this repository
git clone https://github.com/XuHQ1997/simple-tensor.git
cd simple-tensor

# 2. build
mkdir bin
make

# 3. run test
./bin/test
```

You can learn about How to use these code in `test.cpp` .

##### 2. Train a MLP on MNIST

To train a MLP on MNIST:

```shell
# 1. download MNIST dataset from
# 		http://yann.lecun.com/exdb/mnist/

# 2. change dataset path in train_mlp.cpp
#     st::data::MNIST train_dataset(
#        "PATH/TO/MNIST/train-images.idx3-ubyte",
#        "PATH/TO/MNIST/train-labels.idx1-ubyte",
#        batch_size, false
#    );
#    st::data::MNIST val_dataset(
#        "PATH/TO/MNIST/t10k-images.idx3-ubyte",
#        "t10k-labels.idx1-ubyte",
#        batch_size, false
#    );

# 3. build
mkdir bin
make train_mlp

# 4. run train_mlp
./bin/train_mlp
```





