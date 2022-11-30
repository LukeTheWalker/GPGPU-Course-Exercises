CUFLAGS += -arch=sm_86 -O3

%.out: %.cu
	nvcc $< $(CUFLAGS) -o $@

clean:
	rm -f *.o *.out