# Simple-Tensor

Simple-Tensor is trying to implement **Basic operation of Tensor** ,**Autograd based on Dynamic Computational Graph** and **simple Neural Network** in c++.

For many reasons, the repositories such as Pytorch and Tensorflow is too complicated to learn about how Tensors and Computational graphs work. While in this repository, I try to make the concept complete and the code clear.  And of course you shouldn't expect this repo is as efficient as Pytorch or others.

### Features

- [x] Memory management
- [x] Basic operations of Tensor
- [x] Expression Templates
- [ ] Autograd  based on Dynamic Computational Graph
- [ ] Simple Neural Network

### Build and Run

I didn't use any special feature of compiler, so I think it's easy to build. And my development environment is as follows:

```
OS: Windows 10
Compiler: g++ from MinGW-W64-builds-4.3.5
Make: mingw32-make from MinGW-W64-builds-4.3.5
```

To build the project and run test:

``` shell
# 1. clone this repository
git clone https://github.com/XuHQ1997/simple-tensor.git
cd simple-tensor

# 2. make dir of bin
mkdir bin

# 3. build
make

# 4. run
make run
```

You can learn about How to use these code in `test.cpp` .

