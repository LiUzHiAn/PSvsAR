#include "collectives.h"
#include "timer.h"

#include <mpi.h>

#include <stdexcept>
#include <string>
#include <iostream>
#include <vector>
using namespace std;

void TestCollectivesCPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
	// ��ʼ�� CPU 
	InitCollectives(NO_DEVICE);

	// ��ȡ��MPI size �� rank.
	int mpi_size;
	if (MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
		throw std::runtime_error("MPI_Comm_size failed with an error");

	int mpi_rank;
	if (MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
		throw std::runtime_error("MPI_Comm_rank failed with an error");

	timer::Timer timer;
	for (size_t i = 0; i < sizes.size(); i++) {
		auto size = sizes[i];
		auto iters = iterations[i];

		float* data = new float[size];
		float seconds = 0.0f;
		for (size_t iter = 0; iter < iters; iter++) {
			// ��ʼ��ȫΪ1
			for (size_t j = 0; j < size; j++) {
				data[j] = 1.0f;
			}

			float* output;
			timer.start();
			/*********************************/
			/********* �����㷨���� ************/
			RingAllreduce(data, size, &output);
			/********************************/
			/*********************************/

			seconds += timer.seconds();

			// ��֤�������ȷ��.
			for (size_t j = 0; j < size; j++) {
				if (output[j] != (float)mpi_size) {
					std::cerr << "Unexpected result from allreduce: " << data[j] << std::endl;
					return;
				}
			}
			delete[] output;
		}
		if (mpi_rank == 0) {
			std::cout << " allreduce ִ�н����������С -->  "
				<< size
				<< "��ʱ--> "
				<< seconds
				<< " ("
				<< seconds / iters
				<< " per iteration)" << std::endl;
		}

		delete[] data;
	}
}


// baidu-allreduce ����
int main(int argc, char** argv) {
	
	// ��������С  1M,8M,64M,512M
	std::vector<size_t> buffer_sizes = {
		 /*32, 256, 1024, 4096, 16384, 65536, 262144,*/ 1048576, 8388608, 67108864, /*536870912*/
	};

	// ��������������
	/*std::vector<size_t> iterations = {
		100000, 100000, 100000, 100000,
		1000, 1000, 1000, 1000,
		100, 50, 10
	};*/
	std::vector<size_t> iterations = {
		/*100000, 100000, 100000, 100000,*/
		10000, 10000, 10000, /*100000,*/
		/*100000, 100000, 100000*/
	};
	
	TestCollectivesCPU(buffer_sizes, iterations);
	

	// Finalize to avoid any MPI errors on shutdown.
	MPI_Finalize();

	return 0;
}