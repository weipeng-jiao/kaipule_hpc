/* author: weipeng jiao
 * date: 2023-08-16
 * note: runtime api 2
*/


//#include "CL/opencl.h"
#include <CL/cl.h>
// IO stream
#include <iostream>
#include <fstream>
#include <sstream>
#include "stdio.h"

#include<string.h>

using namespace std;


#define ErrInfo(x) printf("\e[1;31m" x "\x1b[0m" "\r\n")

struct GPUInfo
{
	cl_platform_id	Platform = NULL;		// 选择的平台	
	char			*pName = NULL;		// 平台版本名
	cl_uint			uiNumDevices = 0;		// 设备数量
	cl_device_id	*pDevices = NULL;		// 设备
	cl_context		Context = NULL;		// 设备环境
	cl_command_queue CommandQueue = NULL;		// 命令队列
	const char		*pFileName = "add.cl";	// cl文件名
	cl_program		Program = NULL;		// 程序对象
	
	cl_kernel		Kernel = NULL;		// 内核对象
	size_t			uiGlobal_Work_Size[1] = { 0 };	// 用于设定内核分布	
};

#if 0
// runtime api
cl_int clGetPlatformIDs(
                        cl_uint num_entries,    /*设置获取的平台数量*/
                        cl_platform_id *platforms,  /*接收的所有平台信息的指针*/
                        cl_uint *num_platforms);    /*返回当前检测出的平台数量*/


cl_int clGetPlatformInfo(
                        cl_platform_id platform,    /*设置选择的平台*/
                        cl_platform_info param_name,    /*平台信息类型*/
                        size_t param_value_size,    /*所要保存的字节数*/
                        void *param_value,  /*接收信息数据的指针*/
                        size_t *param_value_size_ret);  /*返回实际信息的字节数*/

//设备
cl_int clGetDeviceIDs(    
                        cl_platform_id platform,    /*选择的目标平台*/  
                        cl_device_type device_type,   /*选择的目标平台的设备类型*/
                        cl_uint num_entries,    /*选择设备数量*/
                        cl_device_id *devices,  /*接收所有设备信息的指针*/
                        cl_uint *num_devices);  /*返回总的设备数*/


cl_int clGetDeviceInfo(    
                        cl_device_id device,    /*设置选择的设备*/
                        cl_device_info param_name,  /*平台信息类型*/
                        size_t param_value_size,    /*所要保存的字节数*/
                        void *param_value,  /*信息保存的地址*/
                        size_t *param_value_size_ret);    /*信息实际的字节数*/  

//上下文：创建资源环境
cl_context clCreateContext(
                        cl_context_properties *properties,  /*属性列表*/
                        cl_uint num_devices,  /*设备数量*/
                        const cl_device_id *devices,  /*设备列表*/
                        void *(*pfn_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data),
                        void *user_data, /*提供报错信息*/
                        cl_int *errcode_ret);   /*错误信息大小*/

// 根据设备类型
cl_context clCreateContextFromType(
                        cl_context_properties *properties, 
                        cl_device_type device_type, 
                        void (*pfn_notify)(const char *errinfo, const void *private_info, size_t cb, void *user_data), 
                        void *user_data, 
                        cl_int *errcode_ret);

cl_int clGetContextInfo(    
                        cl_context context, /*上下文*/
                        cl_context_info param_name, /*信息类型*/
                        size_t param_value_size,  /*所要保存的大小*/
                        void *param_value,  /*保存的地址*/
                        size_t param_value_size_ret); /*信息的大小*/

// 命令队列：创建交换管道
cl_command_queue clCreateCommandQueueWithProperties(
                        cl_context context, /*上下文*/
                        cl_device_id device,    /*与上下文关联的设备*/
                        const cl_queue_properties *properties,  /*属性列表*/
                        cl_int *errcode_ret);   /*错误码*/

cl_command_queue clCreateCommandQueue(
                        cl_context context, /*上下文*/
                        cl_device_id device,    /*与上下文关联的设备*/
                        const cl_queue_properties *properties,  /*属性列表*/
                        cl_int *errcode_ret);   /*错误码*/

cl_int clGetCommandQueueInfo(
                        cl_command_queue command_queue, /*命令队列*/
                        cl_command_queue_info parame_name,  /*需要查询的命令队列的属性*/
                        size_t param_value_size, /*保留的字节数*/
                        void *param_value, /*数据指针*/
                        size_t *param_value_size_ret); /*实际有的字节数*/

// 创建程序对象
cl_program clCreateProgramWithSource(
                        cl_context context, /*上下文*/
                        cl_uint count, /*kernel个数*/
                        const char **strings, /*源代码字符串*/
                        const size_t *lengths, /*字符串长度*/
                        cl_int *errcode_ret); /*状态码*/
cl_program clCreateProgramWithBinary(
                        cl_context context,
                        cl_uint num_devices,
                        const cl_device_id *device_list,
                        const size_t *lengths;
                        const unsigned char **binaries,
                        cl_int *binary_status,
                        cl_int *errcode_ret);

// 创建编译对象
cl_int clBuildProgram(
                        cl_program program, /*程序对象*/
                        cl_uint num_deivces,  /*设备数*/
                        const cl_device_id *device_list, /*设备列表*/
                        const char *options, /*编译选项*/
                        void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
                        void *user_data);

cl_int clGetProgramInfo(
                        cl_progpram program, /*程序对象*/
                        cl_program_info param_name, /*信息类型*/
                        size_t param_value_size, /*保留信息长度*/
                        void *param_value, /*数据指针*/
                        size_t *param_value_size_ret); /*实际信息长度*/

cl_int   clGetProgramBuildInfo(
                        cl_program program,  /*程序对象*/
                        cl_device_id device,    /*对应的设备*/
                        cl_program_build_info param_name,  /*信息类型*/
                        size_t param_value_size, /*保留信息长度*/
                        void *param_value, /*数据指针*/
                        size_t *param_value_size_ret); /*实际信息长度*/

// 创建内核对象
cl_kernel clCreateKernel(
                        cl_program program, /*程序对象*/
                        const char *kernel_name, /*调用的kernel*/
                        cl_int *errcode_ret); /*状态码*/

cl_intl clCreateKernelsInProgram(
                        cl_program program, /*程序对象*/
                        cl_uint num_kernels, /*处理的内核个数*/
                        cl_kernel *kernels, /*数据指针*/
                        cl_uint *num_kernels_ret); /*检索程序对象带的内核个数*/

// 创建内存对象
cl_mem clCreateBuffer ( 
                        cl_context context, /*上下文*/
                        cl_mem_flags flags, /*缓冲区的类型*/
                        size_t size, /*内存大小单位字节*/
                        void *host_ptr, /*主机端指针*/
                        cl_int *errcode_ret); /*状态码*/

cl_mem clCreateImage(
                        cl_context context, /*上下文*/
                        cl_mem_flags flags, /*缓冲区的类型*/
                        const cl_image_format *image_format, /*图像的格式*/
                        const cl_image_desc *image_desc, /*图像的类型和维度*/
                        void *host_ptr, /*主机端指针*/
                        cl_int *errcode_ret); /*状态码*/

// 设置内核参数
cl_int clSetKernelArg(
                        cl_kernel kernel, /*内核*/
                        cl_uint arg_index, /*内核参数索引，从左到右，0开始*/
                        size_t arg_size, /*local对应缓冲区大小，global对应类型大小*/
                        const void *arg_value); /*local为NULL，global为buf指针*/

cl_int clGetKernelInfo(
                        cl_kernel kernel, /*内核*/
                        cl_kernel_info param_name, /*查询属性*/
                        size_t param_value_size, /*保存的字节数*/
                        void *param_value, /*数据指针*/
                        size_t *param_value_size_ret); /*实际数据字节数*/

cl_int clGetKernelWorkGroupInfo(
                        cl_kernel kernel, /*内核*/
                        cl_device_id device,  /*设备*/
                        cl_kernel_work_group_info param_name, /*查询属性*/
                        size_t param_value_size, /*保存的字节数*/
                        void *param_value, /*数据指针*/
                        size_t *param_value_size_ret); /*实际数据字节数*/

// 执行内核
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, /*命令队列*/
                                    cl_kernel kernel, /*内核*/
                                    cl_uint work_dim, /*全局工作项的维度*/
                                    const size_t *global_work_offset, /*全局工作项ID的偏移量*/
                                    const size_t *global_work_size, /*全局工作项的大小*/
                                    const sizse_t *local_work_size, /*一个工作组内工作项的大小*/
                                    cl_uint num_events_in_wait_list,/* event_wait_list 内事件的数目*/
                                    const cl_event *event_wait_list,/*需要等待 event_wait_list 内事件执行完成*/
                                    cl_event *event); /*事件对象，查询命令执行状态*/

/* end */
#endif


#if 0
// 编译时从cl文件集成kernel源码
#define STRINGIFY(src) #src 

inline const char* Kernels() {
  static const char* kernels =
    #include "somefile.cl"
    ;
  return kernels;
}
#endif

#if 0
// 编译时从源文件导入多行字符串形式的kernel源码 方式 1
#define KERNEL(...)#__VA_ARGS__

const char *kernels = KERNEL(
                                   __kernel void hellocl(__global uint *buffer)
{
    size_t gidx = get_global_id(0);
    size_t gidy = get_global_id(1);
    size_t lidx = get_local_id(0);
    buffer[gidx + 4 * gidy] = (1 << gidx) | (0x10 << gidy);
 
}
                               );

#endif

#if 0
// 编译时从源文件导入多行字符串形式的kernel源码 方式 2
const char* kernels[]=
{
" __kernel void redution( \n"
" __global int *data, \n"
" __global int *output, \n"
" __local int *data_local \n"
" ) \n"
" { \n"
" int gid=get_group_id(0); \n"
" int tid=get_global_id(0); \n"
" int size=get_local_size(0); \n"
" int id=get_local_id(0); \n"
" data_local[id]=data[tid]; \n"
" barrier(CLK_LOCAL_MEM_FENCE); \n"
" for(int i=size/2;i>0;i>>=1){ \n"
" if(id<i){ \n"
" data_local[id]+=data_local[id+i]; \n"
" } \n"
" barrier(CLK_LOCAL_MEM_FENCE); \n"
" } \n"
" if(id==0){ \n"
" output[gid]=data_local[0]; \n"
" } \n"
" } \n"
};

#endif


#if 1
#define KERNEL(...)#__VA_ARGS__
const char *srcStr = KERNEL(
                        __kernel void add_kernel(__global const float *a,__global const float *b,__global float *result)
                        {
                            int gid = get_global_id(0);
                            result[gid] = hypot(a[gid], b[gid]);
                        }
                            );

#endif

// 获得所有平台信息
void GetPlatformsInfo();
int GetPaltformDevice();
void PrintProfilingInfo(cl_event event);



int main()
{

    GetPlatformsInfo(); // 检索平台信息 如：NVIDIA CUDA、Intel(R) OpenCL HD Graphics、 QUALCOMM Snapdragon(TM)
    GetPaltformDevice();

/* 平台层
1.创建平台
2.创建设备
3.根据设备创建上下文
*/
    #define  ARRAY_SIZE 1280*800*2
    float* src_1=new float[ARRAY_SIZE];
	float* src_2=new float[ARRAY_SIZE];
    float* result=new float[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		src_1[i] = (float)i;
		src_2[i] = (float)(ARRAY_SIZE - i);
	}

   
	cl_int err;
	cl_uint platformNums;
	cl_platform_id PlatformId=NULL;
    cl_device_id device=NULL;
	cl_context context = NULL;
    cl_command_queue commandQueue = NULL;
    cl_program program=NULL;
    //const char**srcStr=NULL;
    //const size_t srcSize = 0;
    cl_kernel kernel = NULL;
    cl_mem memObjects[3]={0};
    cl_event event; 
    

 


    // 1.选择Platform平台
	err = clGetPlatformIDs(1, &PlatformId, &platformNums);  // 选择可用的平台中的第1个
	if (err != CL_SUCCESS || platformNums <= 0)
	{
        ErrInfo("Failed to find any OpenCL platforms !");
		return NULL;
	}

    // 2.选择device设备
    err = clGetDeviceIDs(PlatformId, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // 选择平台的第1个GPU设备
    if (err != CL_SUCCESS)
    {
        ErrInfo("There is no GPU !");
        return NULL;
    }

    // 3.创建上下文
    cl_context_properties properites[] = {CL_CONTEXT_PLATFORM,(cl_context_properties)PlatformId,0}; // 指定使用的平台
    context=clCreateContext(properites,1,&device,NULL,NULL,&err); // 为第1个GPU创建上下文
    if (err != CL_SUCCESS)
    {
        ErrInfo("Greate context err !");
        return NULL;
    }

    // 4.创建命令队列
    commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL); // opencl-1.2生成指令队列的函数 顺序执行 异步执行用opecl-2的接口
    if (commandQueue == NULL)
    {
        ErrInfo("Failed to create commandQueue for device 0");
        return NULL;
    }

    // 5.创建程序对象
    size_t srcSize[] = {strlen(srcStr)};
    program = clCreateProgramWithSource(context, 1,(const char**)&srcStr,srcSize, NULL); // 从字符串中加载1个kernel源代码
    if (program == NULL)
    {
        ErrInfo("Failed to create CL program from source." );
        return NULL;
    }

    // 6.创建编译对象 优化1
    //const char options[] = "-cl-std = CL2.0 -cl-mad-enable -Werror"; // opencl-2.0默认使用opencl-1.2版本，使用2.0版本需要显式的设为CL2.0
    const char options[] = "-cl-std = CL2.0 -cl-mad-enable"; 
    clBuildProgram(program, 1, &device, NULL, NULL, NULL); // 选择设备并编译程序
    if (err != CL_SUCCESS)
    {
        ErrInfo("Failed to build CL program !");
        return NULL;
    }

    // 7.创建内核对象
    kernel = clCreateKernel(program, "add_kernel", &err);
    if (kernel == NULL)
    {
        ErrInfo("Failed to create kernel !");
        return NULL;
    }

    for(int i=0;i<3;i++){
    printf("%d\r\n",i);
    // 8.创建内存对象 copy map ion svm binary
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, src_1, NULL); // 设置第1个数据空间 用作输入
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, src_2, NULL); // 设置第2个数据空间 用作输入
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE, NULL, NULL); // 设置第3个数据空间 用作输出

    // 9.设置内核参数
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);

    // 10.内核入队执行
    size_t globalWorkSize[1] = { ARRAY_SIZE };
	size_t localWorkSize[1] = { 1 };
 

	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &event);

    // 11.读取结果
    err = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, ARRAY_SIZE * sizeof(float), result, 0, NULL, NULL);
	std::cout << result[ARRAY_SIZE-1] <<std::endl;
    
    // 12.读取耗时
    clWaitForEvents(1, &event);
    clFinish(commandQueue);
    PrintProfilingInfo(event);
    }

    // 13.释放资源
    clReleaseMemObject(memObjects[0]); // 释放内存对象 1
    clReleaseMemObject(memObjects[1]); // 释放内存对象 2
    clReleaseMemObject(memObjects[2]); // 释放内存对象 3

    if (commandQueue != 0)
	clReleaseCommandQueue(commandQueue); // 释放命令队列

	if (kernel != 0)
		clReleaseKernel(kernel); // 释放kernel对象
 
	if (program != 0)
		clReleaseProgram(program); // 释放程序对象
 
	if (context != 0)
		clReleaseContext(context); // 释放上下文对象

    delete []src_1;
    delete []src_2;
    delete []result;
    return 0;
  
}





int GetPaltformDevice()
{
    cl_uint numPlatforms = 0;
    cl_platform_id * platforms = nullptr;
    // 第一次调用clGetPlatfromIDs，获取平台数量
    cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if(status != CL_SUCCESS)
    {
        cout << "error : getting platforms failed";
        return 1;
    }
    cout << "FIND " << numPlatforms << " PLATFORM(S)" << endl;
    if(numPlatforms == 0)
        return -1;
    platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    for(int i = 0; i < numPlatforms; ++i)
    {
        // 打印平台信息
        cl_char * param = new cl_char[30];
        cout << "PLATFORM " << i << " INFOMATION :" << endl;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 30, param, nullptr);
        cout << "\tName    : " << param << endl;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 30, param, nullptr);
        cout << "\tVendor  : " << param << endl;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 30, param, nullptr);
        cout << "\tVersion : " << param << endl;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 30, param, nullptr);
        cout << "\tProfile : " << param << endl;
        delete [] param;
 
        // 获取设备
        cl_uint numDevices = 0;
        cl_device_id * devices;
        status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        cout << "PLATFORM " << i << " HAS " << numDevices << " DEVICE(S) : " << endl;
        devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, nullptr);
        // 打印设备信息
        for(int j = 0; j < numDevices; ++j)
        {
            cl_char * device_param = new cl_char[50];
            cout << "DEVICE " << j << " INFOMATION :" << endl;
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 50, device_param, nullptr);
            cout << "\tName    : " << device_param << endl;
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 50, device_param, nullptr);
            cout << "\tVendor  : " << device_param << endl;
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 50, device_param, nullptr);
            cout << "\tVersion : " << device_param << endl;
            clGetDeviceInfo(devices[j], CL_DEVICE_PROFILE, 50, device_param, nullptr);
            cout << "\tProfile : " << device_param << endl;
            delete [] device_param;
        }
        delete [] devices;
        cout << "---------------------------------------" << endl;
    }
    return 0;
}

void GetPlatformsInfo()
{
    // 检索平台信息 如：NVIDIA CUDA、Intel(R) OpenCL HD Graphics、 QUALCOMM Snapdragon(TM)
    cl_uint num_entries;
    clGetPlatformIDs(0, NULL, &num_entries);
    cl_platform_id *platforms = new cl_platform_id[num_entries];
    clGetPlatformIDs(num_entries, platforms, NULL);
    char param_value[512];
    for (cl_uint i = 0; i < num_entries; i++){
        size_t param_value_size_ret;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &param_value_size_ret);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, param_value_size_ret, param_value, NULL);
        printf("platfrom %d name is %s\n", i+1, param_value);
    }
}

void PrintProfilingInfo(cl_event event)
{
    cl_ulong t_queued;
    cl_ulong t_submitted;
    cl_ulong t_started;
    cl_ulong t_ended;
    cl_ulong t_completed;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &t_queued, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &t_submitted, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_started, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_ended, NULL);
#if __linux__
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_COMPLETE, sizeof(cl_ulong), &t_completed, NULL);
#endif

    printf("queue -> submit :" " %fms" "\r\n", (t_submitted - t_queued) * 1e-6);
    printf("submit -> start :" "  %fms" "\r\n", (t_started - t_submitted) * 1e-6);
    printf("start -> end :"  "  %fms"  "\r\n", (t_ended - t_started) * 1e-6);
#if __linux__    
    printf("end -> finish :"  "  %fms"  "\r\n", (t_completed - t_ended) * 1e-6);
    printf("opencl run time :"  "  %fms"  "\r\n", (t_completed - t_submitted) * 1e-6);
#endif

}

