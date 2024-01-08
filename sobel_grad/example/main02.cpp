
//#include "CL/opencl.h"
#include <CL/cl.h>
// IO stream
#include <iostream>
#include <fstream>
#include <sstream>
#include "stdio.h"

#include<string.h>
#include"get_time.h"

using namespace std;


#define ErrInfo(x) printf("\e[1;31m" x "\x1b[0m" "\r\n")


using namespace std;

struct Grad2Int8
{
	signed char x;
	signed char y;
};

// 获得平台、设备和Profiling信息
void GetPlatformsInfo();
int GetPaltformDevice();
void PrintProfilingInfo(cl_event event);

// 数据保存
void save2raw(const char* grad_dst,Grad2Int8* grad_src,int width,int height,int grad_flag);

// 输入输出
#define left_ir_path "./left.raw"
#define right_ir_path "./right.raw"
#define left_grad_x_int8_cl "./left_grad_x_int8_cl.raw"
#define left_grad_y_int8_cl "./left_grad_y_int8_cl.raw"
#define right_grad_x_int8_cl "./right_grad_x_int8_cl.raw"
#define right_grad_y_int8_cl "./right_grad_y_int8_cl.raw"

#if 1
// 导入kernel
#define STRINGIFY(src) #src 
const char* srcStr =
#include "sobel_grad.cl"
;

#endif
int main(int argc, char** argv)
{
    /* --------------------------- 参数配置 --------------------------- */
    int mod_t = 5; 
    int width = 1280;
    int height = 800;
    int memlen =width*height;

    /* --------------------------- 加载数据 --------------------------- */
    
    // 1、创建输入buf
    unsigned char* left_ir_ptr =(unsigned char*)malloc(width*height*sizeof(unsigned char));
    unsigned char* right_ir_ptr =(unsigned char*)malloc(width*height*sizeof(unsigned char));

    // 2、加载测试IR图像
    FILE *img_fp = fopen(argv[1], "rb");
    fseek(img_fp, 0, SEEK_END);
    int bin_size = ftell(img_fp);
    rewind(img_fp);
    if (bin_size!=memlen)
	{
		printf("input image size err\n");
	}

    fread(left_ir_ptr, bin_size, 1, img_fp);
    fclose(img_fp);

    img_fp = fopen(argv[2], "rb");
    fseek(img_fp, 0, SEEK_END);
    bin_size = ftell(img_fp);
    rewind(img_fp);
    if (bin_size!=memlen)
	{
		printf("input image size err\n");
	}

    fread(right_ir_ptr, bin_size, 1, img_fp);
    fclose(img_fp);


    // 3、创建int8类型梯度buf
    Grad2Int8* grad_vec_l_int8 = NULL;
	Grad2Int8* grad_vec_r_int8 = NULL;

	grad_vec_l_int8 = new Grad2Int8[width*height];
	grad_vec_r_int8 = new Grad2Int8[width*height];

	memset(grad_vec_l_int8, 0, width*height * sizeof(Grad2Int8));
	memset(grad_vec_r_int8, 0, width*height * sizeof(Grad2Int8));

    /* --------------------------- OpenCL treatment --------------------------- */

    // 检索平台信息 如：NVIDIA CUDA、Intel(R) OpenCL HD Graphics、 QUALCOMM Snapdragon(TM) 及对应的设备
    GetPlatformsInfo(); 
    GetPaltformDevice();

    // 0.OpenCL变量
	cl_int err;
	cl_uint platformNums;
	cl_platform_id PlatformId=NULL;
    cl_device_id device=NULL;
	cl_context context = NULL;
    cl_command_queue commandQueue = NULL;
    cl_program program=NULL;
    cl_kernel kernel = NULL;
    cl_mem memObjects[4]={0};
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

    // 6.创建编译对象
    const char options[] = "-cl-std=CL2.0 -cl-fast-relaxed-math"; // opencl-2.0默认使用opencl-1.2版本，使用2.0版本需要显式的设为CL2.0

    clBuildProgram(program, 1, &device, options, NULL, NULL); // 选择设备并编译程序
    if (err != CL_SUCCESS)
    {
        ErrInfo("Failed to build CL program !");
        return NULL;
    }

    // 7.创建内核对象
    kernel = clCreateKernel(program, "calc_grad", &err);
    if (kernel == NULL)
    {
        ErrInfo("Failed to create kernel !");
        return NULL;
    }

    // 8.创建内存对象 copy map ion svm binary
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned char) * memlen, NULL, NULL); // 设置第1个数据空间 用作输入
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(unsigned char) * memlen, NULL, NULL); // 设置第2个数据空间 用作输入
    memObjects[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY| CL_MEM_ALLOC_HOST_PTR, sizeof(char) * memlen * 2, NULL, NULL); // 设置第3个数据空间 用作输出
    memObjects[3] = clCreateBuffer(context, CL_MEM_WRITE_ONLY| CL_MEM_ALLOC_HOST_PTR, sizeof(char) * memlen * 2, NULL, NULL); // 设置第4个数据空间 用作输出

    // 9.设置内核参数
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[3]);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &mod_t);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 6, sizeof(int), &height);

    // 10.内核入队执行
    size_t g_work_size[2] = {(size_t)width, (size_t)height};
    size_t l_work_size[2] = {16, 16};    //适配1280*800

    TINIT;  
    for(int i=0;i<10;i++)
    {
        // 模拟新的ir传入GPU
        TIC;
        err = clEnqueueWriteBuffer(commandQueue, memObjects[0], CL_TRUE, 0, sizeof(unsigned char) * memlen,left_ir_ptr,0,NULL,NULL);
        err = clEnqueueWriteBuffer(commandQueue, memObjects[1], CL_TRUE, 0, sizeof(unsigned char) * memlen,right_ir_ptr,0,NULL,NULL);      
        TOC("IR Data Copy to Global Memory");
             
        // 执行kernel
        TIC;
        err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, g_work_size, l_work_size, 0, NULL,&event);
        clWaitForEvents(1, &event);
        clFinish(commandQueue);
        TOC("clEnqueueNDRangeKernel Start to Finish"); 
        PrintProfilingInfo(event);
        printf("\r\n");
    }
  
    TIC;
    // 11.读取结果
    err = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, 2 * memlen * sizeof(char), grad_vec_l_int8, 0, NULL, NULL);
    err = clEnqueueReadBuffer(commandQueue, memObjects[3], CL_TRUE, 0, 2 * memlen * sizeof(char), grad_vec_r_int8, 0, NULL, NULL);
    TOC("copy result form device to host");
 
    
    // 12.结果保存
    save2raw(left_grad_x_int8_cl,(Grad2Int8*)grad_vec_l_int8,width,height,0); // save left float grad used tiff
    save2raw(right_grad_x_int8_cl,(Grad2Int8*)grad_vec_r_int8,width,height,0); // save right float grad used tiff
    save2raw(left_grad_y_int8_cl,(Grad2Int8*)grad_vec_l_int8,width,height,1); // save left float grad used tiff
    save2raw(right_grad_y_int8_cl,(Grad2Int8*)grad_vec_r_int8,width,height,1); // save right float grad used tiff

    // 13.释放资源
    clReleaseMemObject(memObjects[0]); // 释放内存对象 1
    clReleaseMemObject(memObjects[1]); // 释放内存对象 2
    clReleaseMemObject(memObjects[2]); // 释放内存对象 3
    clReleaseMemObject(memObjects[3]); // 释放内存对象 4

    if (commandQueue != 0)
	clReleaseCommandQueue(commandQueue); // 释放命令队列

	if (kernel != 0)
		clReleaseKernel(kernel); // 释放kernel对象
 
	if (program != 0)
		clReleaseProgram(program); // 释放程序对象
 
	if (context != 0)
		clReleaseContext(context); // 释放上下文对象

    delete[] grad_vec_l_int8;
    delete[] grad_vec_r_int8;  
    free(left_ir_ptr);
    free(right_ir_ptr);
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
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_COMPLETE, sizeof(cl_ulong), &t_completed, NULL);

    printf("\e[7m" "Display OpenCl Event Profiling Info" "\x1b[0m" "\r\n");
    printf("queue -> submit :" "\e[1;32m" " %fms" "\x1b[0m" "\r\n", (t_submitted - t_queued) * 1e-6);
    printf("submit -> start :" "\e[1;32m" "  %fms" "\x1b[0m" "\r\n", (t_started - t_submitted) * 1e-6);
    printf("start -> end :" "\e[1;32m" "  %fms" "\x1b[0m" "\r\n", (t_ended - t_started) * 1e-6);
    printf("end -> finish :" "\e[1;32m" "  %fms" "\x1b[0m" "\r\n", (t_completed - t_ended) * 1e-6);
    printf("finish -> queue:" "\e[1;32m" "  %fms" "\x1b[0m" "\r\n", (t_completed - t_queued) * 1e-6);

}

void save2raw(const char* grad_dst,Grad2Int8* grad_src,int width,int height,int grad_flag)
{
    int pixels_num=width*height;
    unsigned char* tmp_ptr =(unsigned char*)malloc(pixels_num*sizeof(unsigned char));
    if(grad_flag==0) // 0 选择保存x方向梯度
    {
        for (int i =0; i < height; i ++)
        {
            for (int j =0; j < width; j ++)
            {
                tmp_ptr[i*width+j]=grad_src[i*width+j].x;
            }
        }
    }
    else if(grad_flag==1) // 1 选择保存y方向梯度
    {
        for (int i =0; i < height; i ++)
        {
            for (int j =0; j < width; j ++)
            {
                tmp_ptr[i*width+j]=grad_src[i*width+j].y;
            }
        }
    }
    else
    {
        printf("grad_flag must be 0 or 1\r\n");
    }

    FILE* fp = fopen(grad_dst, "wb");
    fwrite(tmp_ptr, sizeof(unsigned char), pixels_num, fp);
    fclose(fp);
    free(tmp_ptr);
}

