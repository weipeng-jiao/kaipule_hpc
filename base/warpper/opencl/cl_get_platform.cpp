#include "cl_get_platform.h"


clClassPlatform g_cl_class_inst;
nonClClass g_noncl_class_inst;
//#define INIT_IN_CLASS_CONSTRUCTION
clClassPlatform::clClassPlatform()
{

#ifdef INIT_IN_CLASS_CONSTRUCTION
	// get the number of platforms
	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id *platforms = new cl_platform_id[num_platforms];
	clGetPlatformIDs(num_platforms, platforms, NULL);

	for (cl_uint i = 0; i < num_platforms; i++)
	{
		size_t param_value_size;
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &param_value_size);
		char *platname = new char[param_value_size];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, param_value_size, platname, NULL);
		printf("clClassPlatform-construction: <%d> Platform name is %s\n", i, platname);
		//cout << "<" << i << "> " << "Platform name is :" << platname << endl;
		delete platname;
	}


	delete platforms;
#else
	printf("Construction without cl\n");
#endif

	
}

void clClassPlatform::init(void)
{
#ifndef  INIT_IN_CLASS_CONSTRUCTION
	// get the number of platforms
	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id *platforms = new cl_platform_id[num_platforms];
	clGetPlatformIDs(num_platforms, platforms, NULL);

	for (cl_uint i = 0; i < num_platforms; i++)
	{
		size_t param_value_size;
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &param_value_size);
		char *platname = new char[param_value_size];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, param_value_size, platname, NULL);
		printf("clClassPlatform-init func: <%d> Platform name is %s\n", i, platname);
		//cout << "<" << i << "> " << "Platform name is :" << platname << endl;
		delete platname;
	}


	delete platforms;
	
#endif // ! INIT_IN_CLASS_CONSTRUCTION
}

clClassPlatform::~clClassPlatform()
{

}

nonClClass::nonClClass()
{
	printf("Global nonClClass instance:\n");

}

nonClClass::~nonClClass()
{

}

void clFuncPlatform(void)
{

	// get the number of platforms
	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id *platforms = new cl_platform_id[num_platforms];
	clGetPlatformIDs(num_platforms, platforms, NULL);
	
	for (cl_uint i = 0; i < num_platforms; i++)
	{
		size_t param_value_size;
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &param_value_size);
		char *platname = new char[param_value_size];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, param_value_size, platname, NULL);
		printf("<%d> Platform name is %s\n", i, platname);
		//cout << "<" << i << "> " << "Platform name is :" << platname << endl;
		delete platname;
	}
	

	delete platforms;
    
}

//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platoform ID
//////////////////////////////////////////////////////////////////////////////

cl_int oclGetPlatformID(void)
{

	void* hInst;
	const char *filename;
	//filename = "C:/Windows/SysWOW64/OpenCL.dll";
	filename = "C:/Windows/System32/OpenCL.dll";

	//hInst = LoadLibrary(filename);
	hInst = dlopen(filename, RTLD_LOCAL);
	if (hInst == NULL) {
		//DWORD err = GetLastError();
		printf("Error: %s, when loading %s\n", dlerror(), filename);
	}
	else
	{
		printf("Loading %s Done!\n", filename);
	}
	
	
	void* fp;
	const char *funcname = "clGetPlatformIDs";
	//fp = GetProcAddress((LIBTYPE)hInst, funcname);
	fp = dlsym(hInst, funcname);
	if (fp == NULL) {
		//DWORD err = GetLastError();
		printf("Error: %s, when get func %s\n", dlerror(), fp);
	}
	else
	{
		printf("Loading func %s Done!\n", funcname);
	}

	using clGetPlatformIDsFunc = cl_int(*)(cl_uint, cl_platform_id *, cl_uint *);
	clGetPlatformIDsFunc func = reinterpret_cast<clGetPlatformIDsFunc>(fp);
	cl_uint num_platforms;
	func(0, NULL, &num_platforms); //clGetPlatformIDs
	printf("Number of platforms: %d \n", num_platforms);

	// get platform ID using function
	printf("Functions without Class Instance:\n");
	clFuncPlatform();
	printf("\n");

	// get platform ID using local class instance
	printf("Local clClass Instance:\n");
	clClassPlatform cl_class_inst;
	printf("\n");

	// get platform ID using local static class instance
	printf("Local Static clClass Instance:\n");
	static clClassPlatform static_cl_class_inst;
	printf("\n");

	// get platform ID using global class instance
	printf("Global clClass Instance-init:\n");
	g_cl_class_inst.init();
	printf("\n");

	return CL_SUCCESS;
}