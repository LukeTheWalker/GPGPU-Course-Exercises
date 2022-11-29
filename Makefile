CUFLAGS += -arch=sm_86

%: %.cu
	nvcc $< $(CUFLAGS) -o $@