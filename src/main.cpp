/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <iterator> 
#include <vector>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 24; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *aScan = new int[SIZE];
int* aCompact = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

const int NUM_RUNS = 21;

enum TestCase {
    CPU_SCAN_POT,
    CPU_SCAN_NPOT,
    NAIVE_SCAN_POT,
    NAIVE_SCAN_NPOT,
    EFFICIENT_SCAN_POT,
    EFFICIENT_SCAN_NPOT,
    THRUST_SCAN_POT,
    THRUST_SCAN_NPOT,
    CPU_COMPACT_NOSCAN_POT,
    CPU_COMPACT_NOSCAN_NPOT,
    CPU_COMPACT_SCAN_POT,
    CPU_COMPACT_SCAN_NPOT,
    EFFICIENT_COMPACT_POT,
    EFFICIENT_COMPACT_NPOT,
    LABEL_COUNT
};

constexpr const char* testCaseLabels[] = {
    "cpu scan, power-of-two (std::chrono Measured)",
    "cpu scan, non-power-of-two (std::chrono Measured)",
    "naive scan, power-of-two (CUDA Measured)",
    "naive scan, non-power-of-two (CUDA Measured)",
    "work-efficient scan, power-of-two (CUDA Measured)",
    "work-efficient scan, non-power-of-two (CUDA Measured)",
    "thrust scan, power-of-two (CUDA Measured)",
    "thrust scan, non-power-of-two (CUDA Measured)",
    "cpu compact without scan, power-of-two (std::chrono Measured)",
    "cpu compact without scan, non-power-of-two (std::chrono Measured)",
    "cpu compact with scan, power-of-two (std::chrono Measured)",
    "cpu compact with scan, non-power-of-two (std::chrono Measured)",
    "work-efficient compact, power-of-two (CUDA Measured)",
    "work-efficient compact, non-power-of-two (CUDA Measured)",
};

static_assert(std::size(testCaseLabels) == TestCase::LABEL_COUNT, "label/enum mismatch");

std::array<std::vector<float>, TestCase::LABEL_COUNT> timings;

inline void recordTiming(TestCase t, float ms)
{
    timings[t].push_back(ms);
}

void testCorrectness()
{
    // Scan tests

    printf("\n");
    printf("****************************\n");
    printf("** SCAN CORRECTNESS TESTS **\n");
    printf("****************************\n");
    printf("\n");

    printArray(SIZE, aScan, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, aScan);
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, aScan);
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, aScan);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, aScan);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, aScan);
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, aScan);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, aScan);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, aScan);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, aScan);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************************\n");
    printf("** STREAM COMPACTION CORRECTNESS TESTS **\n");
    printf("*****************************************\n");
    printf("\n");

    // Compaction tests

    printArray(SIZE, aCompact, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, aCompact);
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, aCompact);
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan, power-of-two");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, aCompact);
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithScan(NPOT, c, aCompact);
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, aCompact);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, aCompact);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
}

void testTiming()
{
    printf("\n");
    printf("******************\n");
    printf("** TIMING TESTS **\n");
    printf("******************\n");
    printf("\n");

    for (int i = 0; i < NUM_RUNS; i++)
    {
        // Scan tests

        // initialize b using StreamCompaction::CPU::scan you implement
        // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
        // At first all cases passed because b && c are all zeroes.
        StreamCompaction::CPU::scan(SIZE, b, aScan);
        recordTiming(TestCase::CPU_SCAN_POT, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());

        StreamCompaction::CPU::scan(NPOT, c, aScan);
        recordTiming(TestCase::CPU_SCAN_NPOT, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());

        StreamCompaction::Naive::scan(SIZE, c, aScan);
        recordTiming(TestCase::NAIVE_SCAN_POT, StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());

        StreamCompaction::Naive::scan(NPOT, c, aScan);
        recordTiming(TestCase::NAIVE_SCAN_NPOT, StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());

        StreamCompaction::Efficient::scan(SIZE, c, aScan);
        recordTiming(TestCase::EFFICIENT_SCAN_POT, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());

        StreamCompaction::Efficient::scan(NPOT, c, aScan);
        recordTiming(TestCase::EFFICIENT_SCAN_NPOT, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());

        StreamCompaction::Thrust::scan(SIZE, c, aScan);
        recordTiming(TestCase::THRUST_SCAN_POT, StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());

        StreamCompaction::Thrust::scan(NPOT, c, aScan);
        recordTiming(TestCase::THRUST_SCAN_NPOT, StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());

        // Compaction tests

        // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
        // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
        StreamCompaction::CPU::compactWithoutScan(SIZE, b, aCompact);
        recordTiming(TestCase::CPU_COMPACT_NOSCAN_POT, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());

        StreamCompaction::CPU::compactWithoutScan(NPOT, c, aCompact);
        recordTiming(TestCase::CPU_COMPACT_NOSCAN_NPOT, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());

        StreamCompaction::CPU::compactWithScan(SIZE, c, aCompact);
        recordTiming(TestCase::CPU_COMPACT_SCAN_POT, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());

        StreamCompaction::CPU::compactWithScan(NPOT, c, aCompact);
        recordTiming(TestCase::CPU_COMPACT_SCAN_NPOT, StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());

        StreamCompaction::Efficient::compact(SIZE, c, aCompact);
        recordTiming(TestCase::EFFICIENT_COMPACT_POT, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());

        StreamCompaction::Efficient::compact(NPOT, c, aCompact);
        recordTiming(TestCase::EFFICIENT_COMPACT_NPOT, StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    }

    printf("\n%-66s %12s %12s %12s\n", "test", "median (ms)", "min (ms)", "max (ms)");
    for (int t = 0; t < TestCase::LABEL_COUNT; t++)
    {
        std::vector<float> s = timings[t];
        std::sort(s.begin(), s.end());
        printf("%-66s %12.4f %12.4f %12.4f\n",
            testCaseLabels[t], s[s.size() / 2], s.front(), s.back());
    }
}

int main(int argc, char* argv[])
{
    genArray(SIZE - 1, aScan, 50);  // Leave a 0 at the end to test that edge case
    aScan[SIZE - 1] = 0;

    genArray(SIZE - 1, aCompact, 4);  // Leave a 0 at the end to test that edge case
    aCompact[SIZE - 1] = 0;

    testCorrectness();
    testTiming();

    //system("pause"); // stop Win32 console from closing on exit
    delete[] aScan;
    delete[] aCompact;
    delete[] b;
    delete[] c;
}
