# kaipule_hpc
02 device info

# cuda functions
(1) CHECK(): 检验CUDA函数返回结果状态是否正常
(2) cudaGetDeviceCount(&device_count): 获取当前机器显卡数量
(3) cudaGetDeviceProperties(&prop, i): 获取当前GPU的属性信息
(4) cudaSetDevice(i): 设置使用当前序号的GPU
(5) cudaMemGetInfo( &avail, &total ): avail可使用的GPU显存大小，total显存总的大小

# cuda paramters
(1) prop.name: 获取设备型号
(2) prop.totalGlobalMem: 全部显存大小
(3) total , avail: avail可使用的GPU显存大小，total显存总的大小
(4) prop.major, prop.minor: 架构版本号
(5) prop.totalConstMem: 常量大小
(6) prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]: 网格最大大小
(7) rop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]: block最大
(8) prop.multiProcessorCount: SM个数
(9) prop.sharedMemPerBlock: 每个block的共享内存大小
(10)prop.sharedMemPerMultiprocessor: 每个SM 共享内存大小
(11)prop.regsPerBlock: 每个block中寄存器个数
(12)prop.regsPerMultiprocessor: 每个SM中寄存器个数
(13)prop.maxThreadsPerBlock: 每个block最大的线程数
(14)prop.maxThreadsPerMultiProcessor: 每个SM最大的线程数

