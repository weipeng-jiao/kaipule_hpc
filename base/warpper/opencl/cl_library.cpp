// Copyright 2018 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dlfcn.h>
#include <CL/cl.h>
#include <string>
#include <vector>
#include <iostream>
#include "debug_print.h"

#if __andorid__
#include <CL/cl_ext_qcom.h>
#endif

class cl_library
{
private:
    cl_library();
    ~cl_library();

    bool load();
    void *load_from_path(const std::string &path);
    bool release_handle();
    
    void *m_handle = nullptr;
    
public:
    //~cl_library();

    static cl_library *get();
    
    using clGetPlatformIDsFunc = cl_int (*)(cl_uint, cl_platform_id *, cl_uint *);
    using clGetPlatformInfoFunc =
      cl_int (*)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
    using clBuildProgramFunc = cl_int (*)(cl_program,
                                        cl_uint,
                                        const cl_device_id *,
                                        const char *,
                                        void (*pfn_notify)(cl_program, void *),
                                        void *);
    using clEnqueueNDRangeKernelFunc = cl_int (*)(cl_command_queue,
                                                cl_kernel,
                                                cl_uint,
                                                const size_t *,
                                                const size_t *,
                                                const size_t *,
                                                cl_uint,
                                                const cl_event *,
                                                cl_event *);
    using clSetKernelArgFunc = cl_int (*)(cl_kernel,
                                          cl_uint,
                                          size_t,
                                          const void *);
    using clCreateProgramWithSourceFunc = cl_program (*)(
        cl_context, cl_uint, const char **, const size_t *, cl_int *);
    using clGetProgramBuildInfoFunc = cl_int (*)(cl_program,
                                               cl_device_id,
                                               cl_program_build_info,
                                               size_t,
                                               void *,
                                               size_t *);
    using clReleaseProgramFunc = cl_int (*)(cl_program program);
    using clCreateBufferFunc = 
        cl_mem (*)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
    using clCreateImageFunc =
        cl_mem (*)(cl_context, cl_mem_flags, const cl_image_format *, const cl_image_desc *, void *, cl_int *);
    using clReleaseMemObjectFunc = cl_int (*)(cl_mem);
    using clCreateKernelFunc = cl_kernel (*)(cl_program, const char *, cl_int *);
    using clReleaseKernelFunc = cl_int (*)(cl_kernel kernel);
    using clCreateCommandQueueFunc = cl_command_queue(CL_API_CALL *)(  // NOLINT
      cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
    using clReleaseCommandQueueFunc = cl_int (*)(cl_command_queue);
    using clEnqueueWriteBufferFunc = cl_int (*)(cl_command_queue,
                                              cl_mem,
                                              cl_bool,
                                              size_t,
                                              size_t,
                                              const void *,
                                              cl_uint,
                                              const cl_event *,
                                              cl_event *);
    using clEnqueueReadBufferFunc = cl_int (*)(cl_command_queue,
                                             cl_mem,
                                             cl_bool,
                                             size_t,
                                             size_t,
                                             void *,
                                             cl_uint,
                                             const cl_event *,
                                             cl_event *);
    using clEnqueueReadImageFunc = cl_int (*)(cl_command_queue,
                                            cl_mem,
                                            cl_bool, 
                                            const size_t *,
                                            const size_t *,
                                            size_t,
                                            size_t, 
                                            void *,
                                            cl_uint,
                                            const cl_event *,
                                            cl_event *);
    using clEnqueueWriteImageFunc = cl_int (*)(cl_command_queue,
                                            cl_mem,
                                            cl_bool, 
                                            const size_t *,
                                            const size_t *,
                                            size_t,
                                            size_t, 
                                            const void *,
                                            cl_uint,
                                            const cl_event *,
                                            cl_event *);
    using clEnqueueMapBufferFunc = void *(*)(cl_command_queue,
                                           cl_mem,
                                           cl_bool,
                                           cl_map_flags,
                                           size_t,
                                           size_t,
                                           cl_uint,
                                           const cl_event *,
                                           cl_event *,
                                           cl_int *);
    using clEnqueueUnmapMemObjectFunc = cl_int (*)(
      cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
    using clCreateContextFunc =
        cl_context (*)(const cl_context_properties *,
                        cl_uint,
                        const cl_device_id *,
                        void(CL_CALLBACK *)(  // NOLINT(readability/casting)
                            const char *, const void *, size_t, void *),
                        void *,
                        cl_int *);
    using clReleaseContextFunc = cl_int (*)(cl_context);
    using clGetDeviceInfoFunc =
        cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *);
    using clGetDeviceIDsFunc = cl_int (*)(
        cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
    using clReleaseDeviceFunc = cl_int (*)(cl_device_id);
    using clGetKernelWorkGroupInfoFunc = cl_int (*)(cl_kernel,
                                                  cl_device_id,
                                                  cl_kernel_work_group_info,
                                                  size_t,
                                                  void *,
                                                  size_t *);
    using clFinishFunc = cl_int (*)(cl_command_queue command_queue);
    using clCreateSamplerFunc = cl_sampler (*)(cl_context,
                                            cl_bool,
                                            cl_addressing_mode,
                                            cl_filter_mode,
                                            cl_int *);
    using clReleaseSamplerFunc = cl_int (*)(cl_sampler);
#if __qcom__
    using clSetPerfHintQCOMFunc = cl_int (*)(cl_context, cl_perf_hint);
#endif

    using clWaitForEventsFunc = cl_int (*)(cl_uint, const cl_event *);
    using clReleaseEventFunc = cl_int (*)(cl_event);
    using clGetProgramInfoFunc = cl_int (*)(cl_program,
                                            cl_program_info,
                                            size_t,
                                            void *,
                                            size_t *);
    using clCreateProgramWithBinaryFunc = cl_program (*)(cl_context,
                                                    cl_uint,   
                                                    const cl_device_id *,
                                                    const size_t *,
                                                    const unsigned char **,
                                                    cl_int *,
                                                    cl_int *);            

    using clGetEventProfilingInfoFunc =  cl_int (*)(cl_event,
                                                    cl_profiling_info,
                                                    size_t,
                                                    void*,
                                                    size_t *);


#define CL_DEFINE_FUNC_PTR(func) func##Func func = nullptr;

    CL_DEFINE_FUNC_PTR(clGetPlatformIDs);
    CL_DEFINE_FUNC_PTR(clGetPlatformInfo);
    CL_DEFINE_FUNC_PTR(clBuildProgram);
    CL_DEFINE_FUNC_PTR(clEnqueueNDRangeKernel);
    CL_DEFINE_FUNC_PTR(clSetKernelArg);
    CL_DEFINE_FUNC_PTR(clCreateProgramWithSource);
    CL_DEFINE_FUNC_PTR(clGetProgramBuildInfo);
    CL_DEFINE_FUNC_PTR(clReleaseProgram);
    CL_DEFINE_FUNC_PTR(clCreateBuffer);
    CL_DEFINE_FUNC_PTR(clCreateImage);
    CL_DEFINE_FUNC_PTR(clEnqueueUnmapMemObject);
    CL_DEFINE_FUNC_PTR(clReleaseMemObject);
    CL_DEFINE_FUNC_PTR(clCreateKernel);
    CL_DEFINE_FUNC_PTR(clReleaseKernel);
    CL_DEFINE_FUNC_PTR(clCreateCommandQueue);
    CL_DEFINE_FUNC_PTR(clReleaseCommandQueue);
    CL_DEFINE_FUNC_PTR(clEnqueueMapBuffer);
    CL_DEFINE_FUNC_PTR(clCreateContext);
    CL_DEFINE_FUNC_PTR(clReleaseContext);
    CL_DEFINE_FUNC_PTR(clGetDeviceInfo);
    CL_DEFINE_FUNC_PTR(clGetDeviceIDs);
    CL_DEFINE_FUNC_PTR(clReleaseDevice);
    CL_DEFINE_FUNC_PTR(clGetKernelWorkGroupInfo);
    CL_DEFINE_FUNC_PTR(clFinish);
    CL_DEFINE_FUNC_PTR(clCreateSampler);
    CL_DEFINE_FUNC_PTR(clReleaseSampler);
    CL_DEFINE_FUNC_PTR(clEnqueueReadBuffer);
    CL_DEFINE_FUNC_PTR(clEnqueueWriteBuffer);
    CL_DEFINE_FUNC_PTR(clEnqueueReadImage);
    CL_DEFINE_FUNC_PTR(clEnqueueWriteImage);
    CL_DEFINE_FUNC_PTR(clGetEventProfilingInfo);
#if __qcom__
    CL_DEFINE_FUNC_PTR(clSetPerfHintQCOM);
#endif

    CL_DEFINE_FUNC_PTR(clWaitForEvents);
    CL_DEFINE_FUNC_PTR(clReleaseEvent);
    CL_DEFINE_FUNC_PTR(clGetProgramInfo);
    CL_DEFINE_FUNC_PTR(clCreateProgramWithBinary);

#undef CL_DEFINE_FUNC_PTR

};

cl_library::cl_library()
{
    Debug_log("cl_library start\n");
    this->load();
}

cl_library::~cl_library()
{
    this->release_handle();
    Debug_log("cl_library end\n");
}

bool cl_library::release_handle()
{
  if(m_handle)
  {
    int err = dlclose(m_handle);
    if(err)
    {
        Debug_info("Error %d: release handle failed\n", err);
        return err;
    }
    Debug_info("Release handle success\n", err);
    m_handle = nullptr;
  }

  return true;
}

cl_library *cl_library::get()
{
    static cl_library library;
    return &library;
}

bool cl_library::load()
{
    if(m_handle != nullptr)
    {
        return true;
    }

    // Add customized OpenCL search path here
    // For Qualcomm Adreno with Android
    const std::vector<std::string> paths = {
#if __linux__

#if defined(__aarch64__)
        "/system/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
#else
        "/system/vendor/lib/libOpenCL.so",
        "/system/lib/libOpenCL.so",
#endif

#elif defined(_WIN32)
		"C:/Windows/System32/OpenCL.dll",
#endif
    };

    for(const auto &path : paths)
    {
        void *handle = load_from_path(path);
        if(handle != nullptr)
        {
            m_handle = handle;
			// std::cout << " Load OpenCL.dll from path: " << path << std::endl;
			Debug_log("Load OpenCL.dll from path: %s\n", path.c_str());
            break;
        }
    }

    if(m_handle == nullptr)
    {
        std::cerr << "Error " << dlerror() << " Failed to load OpenCL library." << "\n";
        return false;
    }

    return true;
}

void *cl_library::load_from_path(const std::string &path)
{
    void *handle = dlopen(path.c_str(), RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "Failed to load OpenCL library from path " << path
                << " error code: " << dlerror();
        return nullptr;
    }

#define CL_FUNC_ASSIGN_FROM_DLSYM(func)                                 \
    do {                                                                \
        void *ptr = dlsym(handle, #func);                               \
        if (ptr == nullptr) {                                           \
            std::cerr << "Failed to load" << #func << " from" << path;  \
            continue;                                                   \
        }                                                               \
        func = reinterpret_cast<func##Func>(ptr);                       \
        /*std::cerr << "Loaded " << #func << " from " << path;*/          \
    } while (false)                                                     \

    CL_FUNC_ASSIGN_FROM_DLSYM(clGetPlatformIDs);
    CL_FUNC_ASSIGN_FROM_DLSYM(clGetPlatformInfo);
    CL_FUNC_ASSIGN_FROM_DLSYM(clBuildProgram);
    CL_FUNC_ASSIGN_FROM_DLSYM(clEnqueueNDRangeKernel);
    CL_FUNC_ASSIGN_FROM_DLSYM(clSetKernelArg);
    CL_FUNC_ASSIGN_FROM_DLSYM(clCreateProgramWithSource);
    CL_FUNC_ASSIGN_FROM_DLSYM(clGetProgramBuildInfo);
    CL_FUNC_ASSIGN_FROM_DLSYM(clReleaseProgram);
    CL_FUNC_ASSIGN_FROM_DLSYM(clCreateBuffer);
    CL_FUNC_ASSIGN_FROM_DLSYM(clCreateImage);
    CL_FUNC_ASSIGN_FROM_DLSYM(clEnqueueUnmapMemObject);
    CL_FUNC_ASSIGN_FROM_DLSYM(clReleaseMemObject);
    CL_FUNC_ASSIGN_FROM_DLSYM(clCreateKernel);
    CL_FUNC_ASSIGN_FROM_DLSYM(clReleaseKernel);
    CL_FUNC_ASSIGN_FROM_DLSYM(clCreateCommandQueue);
    CL_FUNC_ASSIGN_FROM_DLSYM(clReleaseCommandQueue);
    CL_FUNC_ASSIGN_FROM_DLSYM(clEnqueueMapBuffer);
    CL_FUNC_ASSIGN_FROM_DLSYM(clCreateContext);
    CL_FUNC_ASSIGN_FROM_DLSYM(clReleaseContext);
    CL_FUNC_ASSIGN_FROM_DLSYM(clGetDeviceInfo);
    CL_FUNC_ASSIGN_FROM_DLSYM(clGetDeviceIDs);
    CL_FUNC_ASSIGN_FROM_DLSYM(clReleaseDevice);
    CL_FUNC_ASSIGN_FROM_DLSYM(clGetKernelWorkGroupInfo);
    CL_FUNC_ASSIGN_FROM_DLSYM(clFinish);
    CL_FUNC_ASSIGN_FROM_DLSYM(clCreateSampler);
    CL_FUNC_ASSIGN_FROM_DLSYM(clReleaseSampler);
    CL_FUNC_ASSIGN_FROM_DLSYM(clEnqueueReadBuffer);
    CL_FUNC_ASSIGN_FROM_DLSYM(clEnqueueWriteBuffer);
    CL_FUNC_ASSIGN_FROM_DLSYM(clEnqueueReadImage);
    CL_FUNC_ASSIGN_FROM_DLSYM(clEnqueueWriteImage);
    CL_FUNC_ASSIGN_FROM_DLSYM(clGetEventProfilingInfo);
#if __qcom__
    CL_FUNC_ASSIGN_FROM_DLSYM(clSetPerfHintQCOM);
#endif

    CL_FUNC_ASSIGN_FROM_DLSYM(clWaitForEvents);
    CL_FUNC_ASSIGN_FROM_DLSYM(clReleaseEvent);
    CL_FUNC_ASSIGN_FROM_DLSYM(clGetProgramInfo);
    CL_FUNC_ASSIGN_FROM_DLSYM(clCreateProgramWithBinary);

#undef CL_FUNC_ASSIGN_FROM_DLSYM

    return handle;
}


/* Platform API */
CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint num_entries,
                                                 cl_platform_id *platforms,
                                                 cl_uint *num_platforms) CL_API_SUFFIX__VERSION_1_0
{
    auto func = cl_library::get()->clGetPlatformIDs;
    if (func != nullptr) {
		Debug_info("clGetPlatformIDs function using cl_library...\n");
        return func(num_entries, platforms, num_platforms);
    } else {
        return CL_INVALID_PLATFORM;
    }
}

CL_API_ENTRY cl_int clGetPlatformInfo(cl_platform_id platform,
                                      cl_platform_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clGetPlatformInfo;
  if (func != nullptr) {
    return func(platform, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Device APIs
CL_API_ENTRY cl_int clGetDeviceIDs(cl_platform_id platform,
                                   cl_device_type device_type,
                                   cl_uint num_entries,
                                   cl_device_id *devices,
                                   cl_uint *num_devices)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clGetDeviceIDs;
  if (func != nullptr) {
    return func(platform, device_type, num_entries, devices, num_devices);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clGetDeviceInfo(cl_device_id device,
                                    cl_device_info param_name,
                                    size_t param_value_size,
                                    void *param_value,
                                    size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clGetDeviceInfo;
  if (func != nullptr) {
    return func(device, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseDevice(cl_device_id device)
    CL_API_SUFFIX__VERSION_1_2 {
  auto func = cl_library::get()->clReleaseDevice;
  if (func != nullptr) {
    return func(device);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Context APIs
CL_API_ENTRY cl_context clCreateContext(
    const cl_context_properties *properties,
    cl_uint num_devices,
    const cl_device_id *devices,
    void(CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
    void *user_data,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clCreateContext;
  if (func != nullptr) {
    return func(properties, num_devices, devices, pfn_notify, user_data,
                errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clReleaseContext(cl_context context)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clReleaseContext;
  if (func != nullptr) {
    return func(context);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Program Object APIs
CL_API_ENTRY cl_program clCreateProgramWithSource(cl_context context,
                                                  cl_uint count,
                                                  const char **strings,
                                                  const size_t *lengths,
                                                  cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clCreateProgramWithSource;
  if (func != nullptr) {
    return func(context, count, strings, lengths, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clGetProgramBuildInfo(cl_program program,
                                          cl_device_id device,
                                          cl_program_build_info param_name,
                                          size_t param_value_size,
                                          void *param_value,
                                          size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clGetProgramBuildInfo;
  if (func != nullptr) {
    return func(program, device, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseProgram(cl_program program)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clReleaseProgram;
  if (func != nullptr) {
    return func(program);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clBuildProgram(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id *device_list,
    const char *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data) CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clBuildProgram;
  if (func != nullptr) {
	Debug_info("clBuildProgram function using cl_library...\n");
    return func(program, num_devices, device_list, options, pfn_notify,
                user_data);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Kernel Object APIs
CL_API_ENTRY cl_kernel clCreateKernel(cl_program program,
                                      const char *kernel_name,
                                      cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clCreateKernel;
  if (func != nullptr) {
    return func(program, kernel_name, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clReleaseKernel(cl_kernel kernel)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clReleaseKernel;
  if (func != nullptr) {
    return func(kernel);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clSetKernelArg(cl_kernel kernel,
                                   cl_uint arg_index,
                                   size_t arg_size,
                                   const void *arg_value)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clSetKernelArg;
  if (func != nullptr) {
    return func(kernel, arg_index, arg_size, arg_value);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Memory Object APIs
CL_API_ENTRY cl_mem clCreateBuffer(cl_context context,
                                   cl_mem_flags flags,
                                   size_t size,
                                   void *host_ptr,
                                   cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clCreateBuffer;
  if (func != nullptr) {
    return func(context, flags, size, host_ptr, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_mem clCreateImage(cl_context context,
                                  cl_mem_flags flags,
                                  const cl_image_format *image_format,
                                  const cl_image_desc *image_desc,
                                  void *host_ptr,
                                  cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_2 {
  auto func = cl_library::get()->clCreateImage;
  if (func != nullptr) {
    return func(context,
                flags,
                image_format,
                image_desc,
                host_ptr,
                errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clReleaseMemObject(cl_mem memobj)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clReleaseMemObject;
  if (func != nullptr) {
    return func(memobj);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Command Queue APIs
CL_API_ENTRY cl_int clReleaseCommandQueue(cl_command_queue command_queue)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clReleaseCommandQueue;
  if (func != nullptr) {
    return func(command_queue);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Enqueued Commands APIs
CL_API_ENTRY cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
                                        cl_mem buffer,
                                        cl_bool blocking_read,
                                        size_t offset,
                                        size_t size,
                                        void *ptr,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event *event_wait_list,
                                        cl_event *event)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clEnqueueReadBuffer;
  if (func != nullptr) {
    return func(command_queue, buffer, blocking_read, offset, size, ptr,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
                                         cl_mem buffer,
                                         cl_bool blocking_write,
                                         size_t offset,
                                         size_t size,
                                         const void *ptr,
                                         cl_uint num_events_in_wait_list,
                                         const cl_event *event_wait_list,
                                         cl_event *event)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clEnqueueWriteBuffer;
  if (func != nullptr) {
    return func(command_queue, buffer, blocking_write, offset, size, ptr,
                num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clEnqueueReadImage(cl_command_queue      command_queue,
                                        cl_mem               image,
                                        cl_bool              blocking_read, 
                                        const size_t *       origin,
                                        const size_t *       region,
                                        size_t               row_pitch,
                                        size_t               slice_pitch,
                                        void *               ptr,
                                        cl_uint              num_events_in_wait_list,
                                        const cl_event *     event_wait_list,
                                        cl_event *           event)
CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clEnqueueReadImage;
  if (func != nullptr) {
    return func(command_queue, image, blocking_read, origin, region, row_pitch,
                slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clEnqueueWriteImage(cl_command_queue    command_queue,
                                        cl_mem              image,
                                        cl_bool             blocking_write,
                                        const size_t *      origin,
                                        const size_t *      region,
                                        size_t              input_row_pitch,
                                        size_t              input_slice_pitch, 
                                        const void *        ptr,
                                        cl_uint             num_events_in_wait_list,
                                        const cl_event *    event_wait_list,
                                        cl_event *          event)
CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clEnqueueWriteImage;
  if (func != nullptr) {
    return func(command_queue, image, blocking_write, origin, region, input_row_pitch,
                input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY void *clEnqueueMapBuffer(cl_command_queue command_queue,
                                      cl_mem buffer,
                                      cl_bool blocking_map,
                                      cl_map_flags map_flags,
                                      size_t offset,
                                      size_t size,
                                      cl_uint num_events_in_wait_list,
                                      const cl_event *event_wait_list,
                                      cl_event *event,
                                      cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clEnqueueMapBuffer;
  if (func != nullptr) {
    return func(command_queue, buffer, blocking_map, map_flags, offset, size,
                num_events_in_wait_list, event_wait_list, event, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue,
                                            cl_mem memobj,
                                            void *mapped_ptr,
                                            cl_uint num_events_in_wait_list,
                                            const cl_event *event_wait_list,
                                            cl_event *event)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clEnqueueUnmapMemObject;
  if (func != nullptr) {
    return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list,
                event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clGetKernelWorkGroupInfo(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clGetKernelWorkGroupInfo;
  if (func != nullptr) {
    return func(kernel, device, param_name, param_value_size, param_value,
                param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
                                           cl_kernel kernel,
                                           cl_uint work_dim,
                                           const size_t *global_work_offset,
                                           const size_t *global_work_size,
                                           const size_t *local_work_size,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event *event_wait_list,
                                           cl_event *event)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clEnqueueNDRangeKernel;
  if (func != nullptr) {
    return func(command_queue, kernel, work_dim, global_work_offset,
                global_work_size, local_work_size, num_events_in_wait_list,
                event_wait_list, event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Sampler APIs
CL_API_ENTRY cl_int clReleaseSampler(cl_sampler sampler) 
CL_API_SUFFIX__VERSION_1_0 {
      auto func = cl_library::get()->clReleaseSampler;
  if (func != nullptr) {
    return func(sampler);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Flush and Finish APIs
CL_API_ENTRY cl_int clFinish(cl_command_queue command_queue)
    CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clFinish;
  if (func != nullptr) {
    return func(command_queue);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

// Deprecated OpenCL 2.0 APIs
CL_API_ENTRY /*CL_EXT_PREFIX__VERSION_1_2_DEPRECATED*/ cl_command_queue
clCreateCommandQueue(cl_context context,
                     cl_device_id device,
                     cl_command_queue_properties properties,
                     cl_int *errcode_ret)
/* CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED */ {  // NOLINT
  auto func = cl_library::get()->clCreateCommandQueue;
  if (func != nullptr) {
    return func(context, device, properties, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY /*CL_EXT_PREFIX__VERSION_1_2_DEPRECATED*/ cl_sampler
clCreateSampler(cl_context context,
                cl_bool normalized_coords,
                cl_addressing_mode addressing_mode,
                cl_filter_mode filter_mode,
                cl_int *errcode_ret)
/* CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED */ {
  auto func = cl_library::get()->clCreateSampler;
  if (func != nullptr) {
    return func(context, normalized_coords, addressing_mode, filter_mode, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

// OpenCL Qualcomm extention APIs
#if __qcom__
CL_API_ENTRY cl_int
clSetPerfHintQCOM(cl_context    context,
                  cl_perf_hint  perf_hint)
{
    auto func = cl_library::get()->clSetPerfHintQCOM;
    if (func != nullptr) {
        return func(context, perf_hint);
    } else {
        return CL_INVALID_PLATFORM;
    }
}
#endif

// Event Object APIs
CL_API_ENTRY cl_int clWaitForEvents(
    cl_uint num_events, const cl_event *event_list) CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clWaitForEvents;
  if (func != nullptr) {    
    return func(num_events, event_list);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_int clReleaseEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clReleaseEvent;
  if (func != nullptr) {    
    return func(event);
  } else {
    return CL_INVALID_PLATFORM;
  }
}


CL_API_ENTRY cl_int clGetProgramInfo(cl_program program,
                                    cl_program_info program_info,
                                    size_t param_value_size,
                                    void *param_value,
                                    size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clGetProgramInfo;
  if (func != nullptr) {    
    return func(program, program_info, param_value_size, param_value, param_value_size_ret);
  } else {
    return CL_INVALID_PLATFORM;
  }
}

CL_API_ENTRY cl_program clCreateProgramWithBinary(cl_context contex,
                                            cl_uint num_devices,
                                            const cl_device_id *device_list,
                                            const size_t *lengths,
                                            const unsigned char **binaries,
                                            cl_int *binary_status,
                                            cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  auto func = cl_library::get()->clCreateProgramWithBinary;
  if (func != nullptr) {    
    return func(contex, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
  } else {
    if (errcode_ret != nullptr) *errcode_ret = CL_INVALID_PLATFORM;
    return nullptr;
  }
}

CL_API_ENTRY cl_int clGetEventProfilingInfo(cl_event event,
                                            cl_profiling_info   param_name,
                                            size_t              param_value_size,
                                            void *              param_value,
                                            size_t *            param_value_size_ret)CL_API_SUFFIX__VERSION_1_0
{
    auto func = cl_library::get()->clGetEventProfilingInfo;
    if(func != nullptr)
    {
      return func(event, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
      return CL_INVALID_PLATFORM;
    }
}