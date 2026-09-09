#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpsweep(int expectedWrites, int stride, int halfStride, int *data)
        {
            // TODO
            int denseIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (denseIdx >= expectedWrites)
            {
                return;
            }

            int stridedIdx = denseIdx * stride;
            data[stridedIdx + stride - 1] += data[stridedIdx + halfStride - 1];
        }

        __global__ void kernDownsweep(int expectedWrites, int stride, int halfStride, int* data)
        {
            // TODO
            int denseIdx = blockIdx.x * blockDim.x + threadIdx.x;
            if (denseIdx >= expectedWrites)
            {
                return;
            }

            int stridedIdx = denseIdx * stride;
            int leftChildIdx = stridedIdx + halfStride - 1;
            int rightChildIdx = stridedIdx + stride - 1;
            int leftChildVal = data[leftChildIdx];
            data[leftChildIdx] = data[rightChildIdx];
            data[rightChildIdx] += leftChildVal;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            if (n < 1)
            {
                return;
            }

            int *dev_data;
            const int blockSize = 256;
            const size_t intSize = sizeof(int);
            int paddedLen = 1 << ilog2ceil(n);
            size_t dataSize = n * intSize;
            size_t paddedDataSize = paddedLen * intSize;
            cudaMalloc((void**)&dev_data, paddedDataSize);
            checkCUDAError("Failed to cudaMalloc dev_data");
            cudaMemset(dev_data, 0, paddedDataSize);
            checkCUDAError("Failed to cudaMemset dev_data in preparation for Upsweep");
            cudaMemcpy(dev_data, idata, dataSize, cudaMemcpyHostToDevice);
            checkCUDAError("Failed to cudaMemcpy idata to dev_data");

            timer().startGpuTimer();
            // TODO
            int steps = ilog2ceil(paddedLen);
            for (int d = 0; d < steps; d++)
            {
                int halfStride = 1 << d;
                int stride = halfStride << 1;
                int expectedWrites = paddedLen / stride;
                kernUpsweep<<<divup(expectedWrites, blockSize), blockSize>>>(expectedWrites, stride, halfStride, dev_data);
                checkCUDAError("Failed to launch kernUpsweep");
            }

            cudaMemset(&dev_data[paddedLen - 1], 0, intSize);
            checkCUDAError("Failed to cudaMemset dev_data in preparation for Downsweep");

            for (int d = steps - 1; d >= 0; d--)
            {
                int halfStride = 1 << d;
                int stride = halfStride << 1;
                int expectedWrites = paddedLen / stride;
                kernDownsweep<<<divup(expectedWrites, blockSize), blockSize>>>(expectedWrites, stride, halfStride, dev_data);
                checkCUDAError("Failed to launch kernDownsweep");
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_data, dataSize, cudaMemcpyDeviceToHost);
            checkCUDAError("Failed to cudaMemcpy dev_data to odata");

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
