/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef _COMMON_OPENCL_
#define _COMMON_OPENCL_

#define __NO_STD_VECTOR // Use cl::vector instead of STL version

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <commons.h>

// OPEN CL STUFF
extern cl_int clError;
extern cl_platform_id platform_id;
extern cl_event event;
extern cl_device_id device_id;
extern cl_context context;
extern cl_program program;
extern cl_command_queue commandQueue;

extern size_t _ls[2];

inline std::string descriptionOfError(cl_int err) {
	switch (err) {
	case CL_SUCCESS:
		return "Success!";
	case CL_DEVICE_NOT_FOUND:
		return "Device not found.";
	case CL_DEVICE_NOT_AVAILABLE:
		return "Device not available";
	case CL_COMPILER_NOT_AVAILABLE:
		return "Compiler not available";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "Memory object allocation failure";
	case CL_OUT_OF_RESOURCES:
		return "Out of resources";
	case CL_OUT_OF_HOST_MEMORY:
		return "Out of host memory";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "Profiling information not available";
	case CL_MEM_COPY_OVERLAP:
		return "Memory copy overlap";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "Image format mismatch";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "Image format not supported";
	case CL_BUILD_PROGRAM_FAILURE:
		return "Program build failure";
	case CL_MAP_FAILURE:
		return "Map failure";
	case CL_INVALID_VALUE:
		return "Invalid value";
	case CL_INVALID_DEVICE_TYPE:
		return "Invalid device type";
	case CL_INVALID_PLATFORM:
		return "Invalid platform";
	case CL_INVALID_DEVICE:
		return "Invalid device";
	case CL_INVALID_CONTEXT:
		return "Invalid context";
	case CL_INVALID_QUEUE_PROPERTIES:
		return "Invalid queue properties";
	case CL_INVALID_COMMAND_QUEUE:
		return "Invalid command queue";
	case CL_INVALID_HOST_PTR:
		return "Invalid host pointer";
	case CL_INVALID_MEM_OBJECT:
		return "Invalid memory object";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "Invalid image format descriptor";
	case CL_INVALID_IMAGE_SIZE:
		return "Invalid image size";
	case CL_INVALID_SAMPLER:
		return "Invalid sampler";
	case CL_INVALID_BINARY:
		return "Invalid binary";
	case CL_INVALID_BUILD_OPTIONS:
		return "Invalid build options";
	case CL_INVALID_PROGRAM:
		return "Invalid program";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return "Invalid program executable";
	case CL_INVALID_KERNEL_NAME:
		return "Invalid kernel name";
	case CL_INVALID_KERNEL_DEFINITION:
		return "Invalid kernel definition";
	case CL_INVALID_KERNEL:
		return "Invalid kernel";
	case CL_INVALID_ARG_INDEX:
		return "Invalid argument index";
	case CL_INVALID_ARG_VALUE:
		return "Invalid argument value";
	case CL_INVALID_ARG_SIZE:
		return "Invalid argument size";
	case CL_INVALID_KERNEL_ARGS:
		return "Invalid kernel arguments";
	case CL_INVALID_WORK_DIMENSION:
		return "Invalid work dimension";
	case CL_INVALID_WORK_GROUP_SIZE:
		return "Invalid work group size";
	case CL_INVALID_WORK_ITEM_SIZE:
		return "Invalid work item size";
	case CL_INVALID_GLOBAL_OFFSET:
		return "Invalid global offset";
	case CL_INVALID_EVENT_WAIT_LIST:
		return "Invalid event wait list";
	case CL_INVALID_EVENT:
		return "Invalid event";
	case CL_INVALID_OPERATION:
		return "Invalid operation";
	case CL_INVALID_GL_OBJECT:
		return "Invalid OpenGL object";
	case CL_INVALID_BUFFER_SIZE:
		return "Invalid buffer size";
	case CL_INVALID_MIP_LEVEL:
		return "Invalid mip-map level";
	default:
		return "Unknown";
	}
}

#define checkErr(err,name) \
  if (err != CL_SUCCESS) {\
    std::cerr << "ERR: " << std::string(name) << "(";\
    std::cerr << descriptionOfError(err);							\
    std::cerr << ") " << __FILE__ << ":"<< __LINE__ << std::endl;	\
    exit(EXIT_FAILURE);\
  }

inline void checkErrX(cl_int err, const char * name) {
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

void opencl_init(void);
void opencl_clean(void);

#define RELEASE_IN_BUFFER(name)  \
    clError = clReleaseMemObject(name##Buffer);\
	checkErr( clError, "clReleaseMemObject");

#define RELEASE_OUT_BUFFER(name) \
    clError = clEnqueueReadBuffer(commandQueue , name##Buffer, CL_TRUE, 0, name##BufferSize, name##Ptr, 0, NULL, NULL );  \
    checkErr( clError, "clEnqueueReadBuffer");\
    clError = clReleaseMemObject(name##Buffer);\
	checkErr( clError, "clReleaseMemObject");

#define RELEASE_INOUT_BUFFER(name)  \
    clError = clEnqueueReadBuffer(commandQueue , name##Buffer, CL_TRUE, 0, name##BufferSize, name##Ptr, 0, NULL, NULL );  \
    checkErr( clError, "clEnqueueReadBuffer");\
    clError = clReleaseMemObject(name##Buffer);\
	checkErr( clError, "clReleaseMemObject");

#define RELEASE_KERNEL(kernelname)      clError = clReleaseKernel(kernelname);  checkErr( clError, "clReleaseKernel");

#define CREATE_KERNEL(name)   CREATE_KERNELVAR(name,#name);
#define CREATE_KERNELVAR(varname,kernelname)   cl_kernel varname = clCreateKernel(program,kernelname, &clError); checkErr( clError, "clCreateKernel" );

#define CREATE_OUT_BUFFER(name,ptr,size)\
    size_t name##BufferSize    =   size;\
    void * name##Ptr           =  (void *) ptr;\
    cl_mem name##Buffer  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY  , name##BufferSize , NULL , &clError);\
    checkErr( clError, "clCreateBuffer input" );

#define CREATE_IN_BUFFER(name,ptr,size)\
    size_t name##BufferSize    =   size;\
    void * name##Ptr           =  (void *) ptr;\
    cl_mem name##Buffer  = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR  , name##BufferSize , name##Ptr , &clError);\
    checkErr( clError, "clCreateBuffer input" );\


#define CREATE_INOUT_BUFFER(name,ptr,size)\
    size_t name##BufferSize    =   size;\
    void * name##Ptr           =  (void *) ptr;\
    cl_mem name##Buffer  = clCreateBuffer(context,  CL_MEM_READ_WRITE  , name##BufferSize , NULL , &clError);\
    checkErr( clError, "clCreateBuffer input" );\
        clError = clEnqueueWriteBuffer(commandQueue, name##Buffer , CL_TRUE, 0, name##BufferSize , ptr, 0, NULL, NULL);\
    checkErr( clError, "clEnqueueWriteBuffer" ) ;

// Remove the CL_TRUE in the write here?

#endif // _COMMON_OPENCL_
