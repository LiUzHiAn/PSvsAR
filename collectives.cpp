#include <vector>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <iostream>

#include <mpi.h>

#include "collectives.h"
using namespace std;

struct MPIGlobalState {
	// MPI运行的设备, -1 代表 CPU-only.
	int device = -1;

	// Whether the global state (and MPI) has been initialized.
	bool initialized = false;
};

static MPIGlobalState global_state;

// 初始化设备信息
void InitCollectives(int device) {
	if (device < 0) {
		// CPU-only 
		int mpi_error = MPI_Init(NULL, NULL);
		if (mpi_error != MPI_SUCCESS) {
			throw std::runtime_error("MPI_Init failed with an error");
		}

		global_state.device = -1;
	}
	global_state.initialized = true;
}

float* alloc(size_t size) {
	if (global_state.device < 0) {
		return new float[size];
	}
}

void dealloc(float* buffer) {
	if (global_state.device < 0) {
		delete[] buffer;
	}
}


void copy(float* dst, float* src, size_t size) {
	if (global_state.device < 0) {

		std::memcpy((void*)dst, (void*)src, size * sizeof(float));
	}
}

// scatter-reduce过程中的规约操作，将src区域的数据叠加到dst上
void reduce(float* dst, float* src, size_t size) {
	if (global_state.device < 0) {
		// Accumulate values from `src` into `dst` on the CPU.
		for (size_t i = 0; i < size; i++) {
			dst[i] += src[i];
		}
	}
}

// 用MPI中的MPI_Allgather方法将各buffer的数据量聚合，确保数据量大小一致
std::vector<size_t> AllgatherInputLengths(int size, size_t this_rank_length) {
	std::vector<size_t> lengths(size);
	MPI_Allgather(&this_rank_length, 1, MPI_UNSIGNED_LONG,
		&lengths[0], 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
	return lengths;
}

void ParametersServer(float* data, size_t length, float** output_ptr)
{
	// Get MPI size and rank.
	int rank;
	int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (mpi_error != MPI_SUCCESS)
		throw std::runtime_error("MPI_Comm_rank failed with an error");

	int size;
	mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (mpi_error != MPI_SUCCESS)
		throw std::runtime_error("MPI_Comm_size failed with an error");

	// 确保每个进程的数据量大小一致
	std::vector<size_t> lengths = AllgatherInputLengths(size, length);
	for (size_t other_length : lengths) {
		if (length != other_length) {
			throw std::runtime_error("RingAllreduce received different lengths");
		}
	}

	// 输出缓冲区
	float* output = alloc(length);

	// 将输出缓冲区初始化为输入数据
	copy(output, data, length);

	// MPI_all_reduce模拟Parameters Server操作
	// all Reduce,为了简单，只在op中用MAX,按道理，神经网络参数更新是应该求平均的
	MPI_Allreduce(data,output,length,MPI_FLOAT,MPI_MAX, MPI_COMM_WORLD);

	// MPI_Finalize();
}
void RingAllreduce(float* data, size_t length, float** output_ptr) {
	// Get MPI size and rank.
	int rank;
	int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (mpi_error != MPI_SUCCESS)
		throw std::runtime_error("MPI_Comm_rank failed with an error");

	int size;
	mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (mpi_error != MPI_SUCCESS)
		throw std::runtime_error("MPI_Comm_size failed with an error");

	// 确保每个进程的数据量大小一致
	std::vector<size_t> lengths = AllgatherInputLengths(size, length);
	for (size_t other_length : lengths) {
		if (length != other_length) {
			throw std::runtime_error("RingAllreduce received different lengths");
		}
	}

	// 将数据分成N段
	const size_t segment_size = length / size;
	std::vector<size_t> segment_sizes(size, segment_size);

	const size_t residual = length % size;
	for (size_t i = 0; i < residual; ++i) {
		segment_sizes[i]++;
	}

	// 计算每个段的段尾
	std::vector<size_t> segment_ends(size);
	segment_ends[0] = segment_sizes[0];
	for (size_t i = 1; i < segment_ends.size(); ++i) {
		segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
	}

	// 最后一个段的段尾应和数据量一致
	assert(segment_ends[size - 1] == length);

	// 输出缓冲区
	float* output = alloc(length);
	*output_ptr = output;

	// 将输出缓冲区初始化为输入数据
	copy(output, data, length);

	// 开辟临时buffer存即将送来的数据，segment_sizes[0]数据是最多的那个段
	float* buffer = alloc(segment_sizes[0]);

	// 从谁获得chunk，把chunk给谁
	const size_t recv_from = (rank - 1 + size) % size;
	const size_t send_to = (rank + 1) % size;

	MPI_Status recv_status;
	MPI_Request recv_req;
	MPI_Datatype datatype = MPI_FLOAT;
	
	// Step 1：scatter-reduce方法
	for (int i = 0; i < size - 1; i++) {
		int recv_chunk = (rank - i - 1 + size) % size;
		int send_chunk = (rank - i + size) % size;
		// 将要发送的数据段首地址
		float* segment_send = &(output[segment_ends[send_chunk] -
			segment_sizes[send_chunk]]);

		MPI_Irecv(buffer, segment_sizes[recv_chunk],
			datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);

		MPI_Send(segment_send, segment_sizes[send_chunk],
			MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);

		// 将要更新的数据段首地址
		float *segment_update = &(output[segment_ends[recv_chunk] -
			segment_sizes[recv_chunk]]);

		// Wait for recv to complete before reduction
		MPI_Wait(&recv_req, &recv_status);
		// 规约操作
		reduce(segment_update, buffer, segment_sizes[recv_chunk]);
	}

	//Step 2： all-gather步骤
	for (size_t i = 0; i < size_t(size - 1); ++i) {
		int send_chunk = (rank - i + 1 + size) % size;
		int recv_chunk = (rank - i + size) % size;
		// 将要发送的数据段首地址
		float* segment_send = &(output[segment_ends[send_chunk] -
			segment_sizes[send_chunk]]);

		// 将要的得到的数据段首地址
		float* segment_recv = &(output[segment_ends[recv_chunk] -
			segment_sizes[recv_chunk]]);

		// send和recv同步，防止出现死锁现象
		MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
			datatype, send_to, 0, segment_recv,
			segment_sizes[recv_chunk], datatype, recv_from,
			0, MPI_COMM_WORLD, &recv_status);
	}


	// MPI_Finalize();

	// 释放临时缓冲区
	dealloc(buffer);

	
}
