/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
#ifndef OCL_UTILS_H
#define OCL_UTILS_H

// *********************************************************************
// Utilities specific to OpenCL samples in NVIDIA GPU Computing SDK 
// *********************************************************************


// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif 

// Includes
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


// reminders for build output window and log
#ifdef _WIN32
#include <windows.h> 
#include "dlfcn.h"
#ifndef EXPORT_DLL
	#define EXPORT_DLL __declspec(dllexport)
#endif // !EXPORT_DLL
#else
#include <dlfcn.h>
	#define EXPORT_DLL 
#endif


class clClassPlatform {
public:
	clClassPlatform();
	~clClassPlatform();
	void init(void);
private:

};

class nonClClass {
public:
	nonClClass();
	~nonClClass();
private:

};


//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default to platform 0
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platform ID
//////////////////////////////////////////////////////////////////////////////
extern "C" EXPORT_DLL cl_int oclGetPlatformID(void);

#endif

