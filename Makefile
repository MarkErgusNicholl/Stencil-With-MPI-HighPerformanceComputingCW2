stencil_MPI: stencil_MPI.c
	mpiicc -std=c11 -Ofast -xAVX -DNOALIAS -w -simd -DALIGNED -ipo $^ -o $@