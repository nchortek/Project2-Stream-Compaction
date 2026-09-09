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

        inline void scan_(int paddedLen, int blockSize, int *dev_data)
        {
            int steps = ilog2ceil(paddedLen);
            for (int d = 0; d < steps; d++)
            {
                int halfStride = 1 << d;
                int stride = halfStride << 1;
                int expectedWrites = paddedLen / stride;
                kernUpsweep<<<divup(expectedWrites, blockSize), blockSize>>>(expectedWrites, stride, halfStride, dev_data);
                checkCUDAError("Failed to launch kernUpsweep");
            }

            cudaMemset(&dev_data[paddedLen - 1], 0, sizeof(int));
            checkCUDAError("Failed to cudaMemset dev_data in preparation for Downsweep");

            for (int d = steps - 1; d >= 0; d--)
            {
                int halfStride = 1 << d;
                int stride = halfStride << 1;
                int expectedWrites = paddedLen / stride;
                kernDownsweep<<<divup(expectedWrites, blockSize), blockSize>>>(expectedWrites, stride, halfStride, dev_data);
                checkCUDAError("Failed to launch kernDownsweep");
            }
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
            int paddedLen = 1 << ilog2ceil(n);
            size_t dataSize = n * sizeof(int);
            size_t paddedDataSize = paddedLen * sizeof(int);
            cudaMalloc((void**)&dev_data, paddedDataSize);
            checkCUDAError("Failed to cudaMalloc dev_data");
            cudaMemset(dev_data, 0, paddedDataSize);
            checkCUDAError("Failed to cudaMemset dev_data in preparation for Upsweep");
            cudaMemcpy(dev_data, idata, dataSize, cudaMemcpyHostToDevice);
            checkCUDAError("Failed to cudaMemcpy idata to dev_data");

            timer().startGpuTimer();
            // TODO
            scan_(paddedLen, blockSize, dev_data);

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
        int compact(int n, int *odata, const int *idata)
        {
            if (n < 1)
            {
                return 0;
            }

            int* dev_idata;
            int* dev_odata;
            int* dev_scan;
            int* dev_mask;
            
            const int blockSize = 256;
            int paddedLen = 1 << ilog2ceil(n);
            size_t dataSize = n * sizeof(int);
            size_t paddedDataSize = paddedLen * sizeof(int);

            cudaMalloc((void**)&dev_scan, paddedDataSize);
            checkCUDAError("Failed to cudaMalloc dev_scan");
            cudaMemset(dev_scan, 0, paddedDataSize);
            checkCUDAError("Failed to cudaMemset dev_scan");

            cudaMalloc((void**)&dev_idata, dataSize);
            checkCUDAError("Failed to cudaMalloc dev_idata");
            cudaMemcpy(dev_idata, idata, dataSize, cudaMemcpyHostToDevice);
            checkCUDAError("Failed to cudaMemcpy idata to dev_idata");

            cudaMalloc((void**)&dev_odata, dataSize);
            checkCUDAError("Failed to cudaMalloc dev_odata");

            cudaMalloc((void**)&dev_mask, dataSize);
            checkCUDAError("Failed to cudaMalloc dev_mask");

            timer().startGpuTimer();
            // TODO
            int numBlocks = divup(n, blockSize);
            StreamCompaction::Common::kernMapToBoolean<<<numBlocks, blockSize>>>(n, dev_mask, dev_idata);
            checkCUDAError("Failed to launch kernMapToBoolean");

            cudaMemcpy(dev_scan, dev_mask, dataSize, cudaMemcpyDeviceToDevice);
            checkCUDAError("Failed to cudaMemcpy dev_mask to dev_scan");

            scan_(paddedLen, blockSize, dev_scan);

            StreamCompaction::Common::kernScatter<<<numBlocks, blockSize>>>(n, dev_odata, dev_idata, dev_mask, dev_scan);
            checkCUDAError("Failed to launch kernScatter");
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, dataSize, cudaMemcpyDeviceToHost);
            checkCUDAError("Failed to cudaMemcpy dev_odata to odata");

            int exclusiveCount;
            cudaMemcpy(&exclusiveCount, &dev_scan[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Failed to cudaMemcpy exclusiveCount");

            int inclusiveMask;
            cudaMemcpy(&inclusiveMask, &dev_mask[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Failed to cudaMemcpy inclusiveMask");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_scan);
            cudaFree(dev_mask);

            return exclusiveCount + inclusiveMask;
        }
    }
}
