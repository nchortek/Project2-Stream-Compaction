#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int stride, int *odata, const int *idata)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
            {
                return;
            }

            if (idx < stride)
            {
                odata[idx] = idata[idx];
            }
            else
            {
                odata[idx] = idata[idx] + idata[idx - stride];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            if (n < 1)
            {
                return;
            }

            odata[0] = 0;
            
            if (n == 1)
            {
                return;
            }

            int* dev_data1;
            int* dev_data2;
            const size_t intSize = sizeof(int);
            const int blockSize = 256;
            const int numBlocks = divup(n, blockSize);
            size_t dataSize = n * intSize;
            cudaMalloc((void**)&dev_data1, dataSize);
            checkCUDAError("Failed to cudaMalloc dev_data1");
            cudaMalloc((void**)&dev_data2, dataSize);
            checkCUDAError("Failed to cudaMalloc dev_data2");
            cudaMemcpy(dev_data1, idata, dataSize, cudaMemcpyHostToDevice);
            checkCUDAError("Failed to cudaMemcpy dev_data1");

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++)
            {
                int stride = 1 << (d - 1);
                kernNaiveScan<<<numBlocks, blockSize>>>(n, stride, dev_data2, dev_data1);
                checkCUDAError("Failed to launch kernNaiveScan");
                std::swap(dev_data1, dev_data2);
            }
            timer().endGpuTimer();

            // deviceToHost copy dev_padded to odata (n copies)
            cudaMemcpy(&odata[1], dev_data1, dataSize - intSize, cudaMemcpyDeviceToHost);
            checkCUDAError("Failed to copy dev_data1 to odata");
            cudaFree(dev_data1);
            cudaFree(dev_data2);
        }
    }
}
