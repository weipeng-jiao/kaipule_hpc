#include <stdio.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <device_launch_parameters.h>

//定义全局宏 判断结果返回是否异常
#define CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int args, char ** argv) 
{
    int device_count = 0;
    //获取当前机器显卡数量
    CHECK(cudaGetDeviceCount(&device_count));
    printf("Device count %d\n", device_count);

   //遍历当前所有的卡，获取其属性信息
    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp prop;
        //获取当前GPU的属性信息
        CHECK(cudaGetDeviceProperties(&prop, i));
        //设置使用当前序号的GPU
        cudaSetDevice(i);
        //avail可使用的GPU显存大小，total显存总的大小
        size_t avail;
        size_t total;
        cudaMemGetInfo( &avail, &total ); 

        printf("Device name %s\n", prop.name);
        //全部显存大小
        printf("Amount of global memory: %g GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        //全部显存及剩余可用显存
        printf("Amount of total memory: %g GB avail memory: %g \n", total / (1024.0 * 1024.0 * 1024.0), avail / (1024.0 * 1024.0 * 1024.0));
        //计算能力：标识设备的核心架构、gpu硬件支持的功能和指令，有时也被称为“SM version”
        printf("Compute capability:     %d.%d\n", prop.major, prop.minor);
        //常量大小
        printf("Amount of constant memory:      %g KB\n", prop.totalConstMem / 1024.0);
        //网格最大大小
        printf("Maximum grid size:  %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        //block最大
        printf("maximum block size:     %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        //SM个数
        printf("Number of SMs:      %d\n", prop.multiProcessorCount);
        //每个block的共享内存大小
        printf("Maximum amount of shared memory per block: %g KB\n", prop.sharedMemPerBlock / 1024.0);
        //每个SM 共享内存大小
        printf("Maximum amount of shared memory per SM:    %g KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
        //每个block中寄存器个数
        printf("Maximum number of registers per block:     %d K\n", prop.regsPerBlock / 1024);
        //每个SM中寄存器个数
        printf("Maximum number of registers per SM:        %d K\n", prop.regsPerMultiprocessor / 1024);
        //每个block最大的线程数
        printf("Maximum number of threads per block:       %d\n", prop.maxThreadsPerBlock);
        //每个SM最大的线程数
        printf("Maximum number of threads per SM:          %d\n", prop.maxThreadsPerMultiProcessor);
    }
}