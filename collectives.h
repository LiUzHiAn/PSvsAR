#ifndef BAIDU_ALLREDUCE_COLLECTIVES_H_
#define BAIDU_ALLREDUCE_COLLECTIVES_H_ value

#include <cstddef>

#include <mpi.h>

#define NO_DEVICE -1

/*
* baidu-allreduce communication collectives, and provides the following functions:
*/

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
void InitCollectives(int device);

// The ring allreduce. The lengths of the data chunks passed to this function
// must be the same across all MPI processes. The output memory will be
// allocated and written into `output`.
void RingAllreduce(float* data, size_t length, float** output);


#endif 