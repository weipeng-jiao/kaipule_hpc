/* author: weipeng jiao
 * date: 2023-08-16
 * note: Get Platforms and Device Info 2
*/
#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <iostream>  
#include <CL/cl.h>  
  
  
int main() {  
    system("chcp 65001");  
    //cl_platform 表示一个OpenCL的执行平台，关联到GPU硬件，如N卡，AMD卡  
    cl_platform_id *platforms;  
  
    //OpenCL中定义的跨平台的usigned int和int类型  
    cl_uint num_platforms;  
    cl_int i, err, platform_index = -1;  
  
    char *ext_data;  
    size_t ext_size;  
    const char icd_ext[] = "cl_khr_icd";  
  
    //要使platform工作，需要两个步骤。1 需要为cl_platform_id结构分配内存空间。2 需要调用clGetPlatformIDs初始化这些数据结构。一般还需要步骤0：询问主机上有多少platforms  
  
    //查询计算机上有多少个支持OpenCL的设备  
    err = clGetPlatformIDs(5, NULL, &num_platforms);  
    if (err < 0) {  
        perror("Couldn't find any platforms.");  
        exit(1);  
    }  
    printf("本机上支持OpenCL的环境数量: %d\n", num_platforms);  
  
    //为platforms分配空间  
    platforms = (cl_platform_id *)  
            malloc(sizeof(cl_platform_id) * num_platforms);  
  
    clGetPlatformIDs(num_platforms, platforms, NULL);  
  
    //获取GPU平台的详细信息  
    for (i = 0; i < num_platforms; i++) {  
        //获取缓存大小  
        err = clGetPlatformInfo(platforms[i],  
                                CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);  
        if (err < 0) {  
            perror("Couldn't read extension data.");  
            exit(1);  
        }  
  
        printf("缓存大小: %zd\n", ext_size);  
  
        ext_data = (char *) malloc(ext_size);  
  
        //获取支持的扩展功能  
        clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS,  
                          ext_size, ext_data, NULL);  
        printf("平台 %d 支持的扩展功能: %s\n", i, ext_data);  
  
        //获取显卡的名称  
        char *name = (char *) malloc(ext_size);  
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,  
                          ext_size, name, NULL);  
        printf("平台 %d 是: %s\n", i, name);  
  
        //获取显卡的生产商名称  
        char *vendor = (char *) malloc(ext_size);  
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,  
                          ext_size, vendor, NULL);  
        printf("平台 %d 的生产商是: %s\n", i, vendor);  
  
        //获取平台版本  
        char *version = (char *) malloc(ext_size);  
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION,  
                          ext_size, version, NULL);  
        printf("平台 %d 的版本信息： %s\n", i, version);  
  
        //查询显卡是独立的还是嵌入的  
        char *profile = (char *) malloc(ext_size);  
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE,  
                          ext_size, profile, NULL);  
        printf("平台 %d 是独立的(full profile)还是嵌入式的(embeded profile)?: %s\n", i, profile);  
  
        //查询是否支持ICD扩展  
        if (strstr(ext_data, icd_ext) != NULL)  
            platform_index = i;  
        std::cout << "平台ID = " << platform_index << std::endl;  
        /* Display whether ICD extension is supported */  
        if (platform_index > -1)  
            printf("平台 %d 支持ICD扩展: %s\n",  
                   platform_index, icd_ext);  
        std::cout << std::endl;  
  
        //释放空间  
        free(ext_data);  
        free(name);  
        free(vendor);  
        free(version);  
        free(profile);  
    }  
  
    if (platform_index <= -1)  
        printf("No platforms support the %s extension.\n", icd_ext);  
    getchar();  
  
    //释放资源  
    free(platforms);  
    return 0;  
}
