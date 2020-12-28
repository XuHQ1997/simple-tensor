CXX := g++
CXX_FLAGS := -std=c++11 -O2

BIN := bin
INCLUDE := include
SRC := src

folders = utils exp exp/operator tensor nn data
all_header_files  = $(foreach folder, $(folders), $(wildcard $(INCLUDE)/$(folder)/*.h))
all_src_files     = $(foreach folder, $(folders), $(wildcard $(SRC)/$(folder)/*.cpp))
all_src_basenames = $(basename $(notdir $(all_src_files)))
all_objects       = $(addprefix $(BIN)/, $(addsuffix .o, $(all_src_basenames)))

.PHONY : test train_mlp train_cnn clean

test: $(BIN)/test.exe
	@echo build test finished

train_mlp: $(BIN)/train_mlp.exe
	@echo build train_mlp finished

train_cnn: $(BIN)/train_cnn.exe
	@echo build train_cnn finished

$(BIN)/test.exe: $(BIN)/test.o $(all_objects)
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -o $@ $^

$(BIN)/test.o: test.cpp $(all_header_files)
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $@ $< 

$(BIN)/train_mlp.exe: $(BIN)/train_mlp.o $(all_objects)
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -o $@ $^

$(BIN)/train_mlp.o: train_mlp.cpp $(all_header_files)
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $@ $< 

$(BIN)/train_cnn.exe: $(BIN)/train_cnn.o $(all_objects)
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -o $@ $^

$(BIN)/train_cnn.o: train_cnn.cpp $(all_header_files)
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $@ $< 

clean:
	rm $(BIN)/*.o
	rm $(BIN)/*.exe

# The following content is automatically generated by update_makefile.py


$(BIN)/data.o: src\data\data.cpp include/data/data.h include/utils/base_config.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/data.o src\data\data.cpp

$(BIN)/init.o: src\nn\init.cpp include/nn/init.h include/utils/exception.h \
 include/tensor/tensor.h include/exp/exp.h include/exp/exp_impl.h \
 include/utils/allocator.h include/utils/base_config.h \
 include/utils/array.h include/exp/grad_impl.h \
 include/exp/operator/log_softmax.h include/exp/operator/constant.h \
 include/exp/operator/reduce_op.h include/exp/operator/nll_loss.h \
 include/exp/operator/conv.h include/exp/operator/basic_op.h \
 include/tensor/tensor_impl.h include/tensor/storage.h \
 include/tensor/shape.h include/tensor/grad_meta.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/init.o src\nn\init.cpp

$(BIN)/module.o: src\nn\module.cpp include/exp/function.h \
 include/utils/allocator.h include/utils/base_config.h \
 include/utils/exception.h include/exp/exp_impl.h include/utils/array.h \
 include/exp/grad_impl.h include/exp/operator/log_softmax.h \
 include/exp/operator/constant.h include/exp/operator/reduce_op.h \
 include/exp/operator/nll_loss.h include/exp/operator/conv.h \
 include/exp/exp.h include/exp/operator/basic_op.h \
 include/exp/operator/matrix_op.h include/nn/module.h \
 include/tensor/tensor.h include/tensor/tensor_impl.h \
 include/tensor/storage.h include/tensor/shape.h \
 include/tensor/grad_meta.h include/nn/init.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/module.o src\nn\module.cpp

$(BIN)/optim.o: src\nn\optim.cpp include/tensor/storage.h \
 include/utils/base_config.h include/utils/allocator.h \
 include/tensor/tensor.h include/exp/exp.h include/exp/exp_impl.h \
 include/utils/array.h include/exp/grad_impl.h include/utils/exception.h \
 include/exp/operator/log_softmax.h include/exp/operator/constant.h \
 include/exp/operator/reduce_op.h include/exp/operator/nll_loss.h \
 include/exp/operator/conv.h include/exp/operator/basic_op.h \
 include/tensor/tensor_impl.h include/tensor/shape.h \
 include/tensor/grad_meta.h include/nn/optim.h include/nn/module.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/optim.o src\nn\optim.cpp

$(BIN)/shape.o: src\tensor\shape.cpp include/tensor/shape.h \
 include/utils/base_config.h include/utils/allocator.h \
 include/utils/array.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/shape.o src\tensor\shape.cpp

$(BIN)/storage.o: src\tensor\storage.cpp include/tensor/storage.h \
 include/utils/base_config.h include/utils/allocator.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/storage.o src\tensor\storage.cpp

$(BIN)/tensor.o: src\tensor\tensor.cpp include/tensor/tensor.h include/exp/exp.h \
 include/exp/exp_impl.h include/utils/allocator.h \
 include/utils/base_config.h include/utils/array.h \
 include/exp/grad_impl.h include/utils/exception.h \
 include/exp/operator/log_softmax.h include/exp/operator/constant.h \
 include/exp/operator/reduce_op.h include/exp/operator/nll_loss.h \
 include/exp/operator/conv.h include/exp/operator/basic_op.h \
 include/tensor/tensor_impl.h include/tensor/storage.h \
 include/tensor/shape.h include/tensor/grad_meta.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/tensor.o src\tensor\tensor.cpp

$(BIN)/tensor_impl.o: src\tensor\tensor_impl.cpp include/tensor/tensor_impl.h \
 include/exp/exp_impl.h include/utils/allocator.h \
 include/utils/base_config.h include/utils/array.h \
 include/exp/grad_impl.h include/utils/exception.h \
 include/exp/operator/log_softmax.h include/exp/operator/constant.h \
 include/exp/operator/reduce_op.h include/exp/operator/nll_loss.h \
 include/exp/operator/conv.h include/tensor/storage.h \
 include/tensor/shape.h include/tensor/grad_meta.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/tensor_impl.o src\tensor\tensor_impl.cpp

$(BIN)/allocator.o: src\utils\allocator.cpp include/utils/allocator.h \
 include/utils/base_config.h include/utils/exception.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/allocator.o src\utils\allocator.cpp

$(BIN)/exception.o: src\utils\exception.cpp include/utils/exception.h
	$(CXX) $(CXX_FLAGS) -I $(INCLUDE) -c -o $(BIN)/exception.o src\utils\exception.cpp

