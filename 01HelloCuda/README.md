# kaipule_hpc
01 HelloCuda

(1)cuda.h: cuda驱动底层库的接口, 链接libcuda.so, 和底层驱动有关的函数接口以cu做前缀
(2)cuda_runtime.h: 将底层库进一次封装的高级应用接口, 方便开发人员开发，链接libcudart.so, 和运行时有关的函数接口以cuda做前缀
(3)cuda_runtime_api.h: 是cuda_runtime.h的子集, cuda_runtime_api.h是纯C接口和实现, 而cuda_runtime.h是C++接口和实现
(4)device_launch_parameters.h: 包含cuda内置变量

(1)__device__ : 为CUDA的关键字,表示代码在设备端(GPU端)运行, 只在CPU端被调用, 与__global__互斥
(2)__global__ : 为CUDA的关键字,表示代码在设备端(GPU端)运行, 可以在CPU端被调用, 与__device__互斥
(3)__host__ : 为CUDA的关键字,表示代码在主机端(CPU端)运行, 只在CPU端被调用, 函数可共同被__device__和__host__修饰

(1)test <<<1, 1 >>> (): 函数调用,  <<< >>>中的第一个参数表示块的个数, 第二个参数表示每个线程块中线程的个数
(2)cudaDeviceSynchronize(): 会阻塞当前程序的执行，直到所有任务都处理完毕