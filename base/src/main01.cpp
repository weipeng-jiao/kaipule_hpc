/* author: weipeng jiao
 * date: 2023-08-16
 * note: Hello OpenCL
*/

#include <iostream>
#include <string>
// opencl
#include "CL\opencl.h"
using namespace std;

int main()
{
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

    printf("Hello OpenCL\r\n");
    return 0;
}