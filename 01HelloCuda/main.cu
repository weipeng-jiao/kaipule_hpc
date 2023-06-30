#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
using namespace std;

// __device__ 为CUDA的关键字,表示代码在设备端(GPU端)运行, 只在CPU端被调用
__device__ int helloCuda(void)
{
    printf("Hello CUDA!\n");
    return 0;
}

// __global__ 为CUDA的关键字,表示代码在设备端(GPU端)运行, 可以在CPU端被调用
__global__ void test(void)  
{
    helloCuda();
}

// __host__ 为CUDA的关键字,表示代码在主机端(CPU端)运行, 只在CPU端被调用
__host__ int main()
{
  
    test <<<1, 1 >>> ();  // 函数调用,  <<< >>>中的第一个参数表示块的个数, 第二个参数表示每个线程块中线程的个数
    // 这里是使用一个线程块,这个线程块中只有一个线程执行这个函数.
    cudaDeviceSynchronize(); // 会阻塞当前程序的执行，直到所有任务都处理完毕（这里的任务其实就是指的是所有的线程都已经执行完了kernel function）。
    // 通俗讲,就是等待设备端的线程执行完成
    // 一个线程块中可以有多个线程,GPU的线程是GPU的最小操作单元

    return 0;
}
