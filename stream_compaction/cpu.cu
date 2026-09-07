#include <cassert>
#include <cstdio>
#include <vector>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
        * CPU scan implementation
        */
        inline void scan_(int n, int* odata, const int* idata)
        {
            assert(n >= 1);

            odata[0] = 0;

            for (int i = 1; i < n; i++)
            {
                odata[i] = idata[i - 1] + odata[i - 1];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata)
        {
            if (n < 1)
            {
                return;
            }

            timer().startCpuTimer();
            // TODO
            scan_(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata)
        {
            timer().startCpuTimer();
            // TODO
            int count = 0;

            for (int i = 0; i < n; i++)
            {
                if (int curVal = idata[i]; curVal != 0)
                {
                    odata[count] = curVal;
                    count++;
                }
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata)
        {
            if (n < 1)
            {
                return 0;
            }

            std::vector<int> mask(n);
            std::vector<int> scanOutput(n);

            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; i++)
            {
                mask[i] = idata[i] != 0 ? 1 : 0;
            }

            scan_(n, scanOutput.data(), mask.data());

            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (mask[i] != 0)
                {
                    count++;
                    odata[scanOutput[i]] = idata[i];
                }
            }

            timer().endCpuTimer();
            return count;
        }
    }
}
