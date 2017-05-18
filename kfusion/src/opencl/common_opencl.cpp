/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#define EXTERNS
#include "common_opencl.h"
#include <sstream>

// OPEN CL STUFF
cl_int clError = CL_SUCCESS;
cl_platform_id platform_id = 0;
cl_device_id device_id;             // compute device id 
cl_context context;
cl_program program;
cl_command_queue commandQueue;

void opencl_clean(void) {

	clReleaseContext(context);
	clReleaseCommandQueue(commandQueue);
	clReleaseProgram(program);

	return;
}

void opencl_init(void) {

	// get the platform

	cl_uint num_platforms;
	clError = clGetPlatformIDs(0, NULL, &num_platforms);
	checkErr(clError, "clGetPlatformIDs( 0, NULL, &num_platforms );");

	if (num_platforms <= 0) {
		std::cout << "No platform..." << std::endl;
		exit(1);
	}

	cl_platform_id* platforms = new cl_platform_id[num_platforms];
	clError = clGetPlatformIDs(num_platforms, platforms, NULL);
	checkErr(clError, "clGetPlatformIDs( num_platforms, &platforms, NULL );");

	platform_id = platforms[0];
	if(getenv("OPENCL_PLATFORM")){
		int platform_index = atoi(getenv("OPENCL_PLATFORM"));
		if(platform_index >= 0 && platform_index < num_platforms){
			platform_id = platforms[platform_index];
		} else {
			std::cerr << "Invalid OpenCL Platform Index " << platform_index 
				<< " defaulting to: 0" << std::endl;
		}
	}
	char platformName[256];
	clError = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR,
			sizeof(platformName), platformName, NULL);
	std::cerr << "Using OpenCL Platform: " << platformName
			<< std::endl;

	delete platforms;

	// Connect to a compute device
	//
	cl_uint device_count = 0;
	clError = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL,
			&device_count);
	checkErr(clError, "Failed to create a device group");
	cl_device_id* deviceIds = (cl_device_id*) malloc(
			sizeof(cl_device_id) * device_count);
	clError = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, device_count,
			deviceIds, NULL);
	int deviceIndex = 0;
	if(getenv("OPENCL_DEVICE")){
		int index = atoi(getenv("OPENCL_DEVICE"));
		if(index >= 0 && index < device_count){
			deviceIndex = index;
		} else {
			std::cerr << "Invalid OpenCL Device Index " << index 
				<< " defaulting to: 0" << std::endl;
		}
	}
	char device_name[256];
	int compute_units;
        
	clError = clGetDeviceInfo(deviceIds[deviceIndex], CL_DEVICE_NAME,
			sizeof(device_name), device_name, NULL);
	checkErr(clError, "clGetDeviceInfo failed");
	clError = clGetDeviceInfo(deviceIds[deviceIndex], CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof(cl_uint), &compute_units, NULL);
	checkErr(clError, "clGetDeviceInfo failed");
	std::cerr << "Using OpenCL device  : " << device_name;
	std::cerr << " with " << compute_units << " compute units" << std::endl;
	device_id = deviceIds[deviceIndex];
	delete deviceIds;
	// Create a compute context 
	//
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &clError);
	checkErr(clError, "Failed to create a compute context!");

	// Create a command commands
	//
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	commandQueue = clCreateCommandQueue(context, device_id, 0, &clError);
	checkErr(clError, "Failed to create a command commands!");

	// READ KERNEL FILENAME
	std::string filename = "NOTDEFINED.cl";
	char const* tmp_name = getenv("OPENCL_KERNEL");
	if (tmp_name) {
		filename = std::string(tmp_name);
	} else {
		filename = std::string(__FILE__);
		filename = filename.substr(0, filename.length() - 17);
		filename += "/kernels.cl";

	}

	// READ OPENCL_PARAMETERS
	std::string compile_parameters = "";
	char const* tmp_params = getenv("OPENCL_PARAMETERS");
	if (tmp_params) {
		compile_parameters = std::string(tmp_params);
	}

	std::ifstream kernelFile(filename.c_str(), std::ios::in);

	if (!kernelFile.is_open()) {
		std::cout << "Unable to open " << filename << ". " << __FILE__ << ":"
				<< __LINE__ << "Please set OPENCL_KERNEL" << std::endl;
		exit(1);
	}

	/*
	 * Read the kernel file into an output stream.
	 * Convert this into a char array for passing to OpenCL.
	 */
	std::ostringstream outputStringStream;
	outputStringStream << kernelFile.rdbuf();
	std::string srcStdStr = outputStringStream.str();
	const char* charSource = srcStdStr.c_str();

	kernelFile.close();
	// Create the compute program from the source buffer
	//
	program = clCreateProgramWithSource(context, 1, (const char **) &charSource,
			NULL, &clError);
	if (!program) {
		printf("Error: Failed to create compute program!\n");
		exit(1);
	}

	// Build the program executable
	//
	clError = clBuildProgram(program, 0, NULL, compile_parameters.c_str(), NULL,
			NULL);

	/* Get the size of the build log. */
	size_t logSize = 0;
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
			&logSize);

	if (clError != CL_SUCCESS) {
		if (logSize > 1) {
			char* log = new char[logSize];
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
					logSize, log, NULL);

			std::string stringChars(log, logSize);
			std::cerr << "Build log:\n " << stringChars << std::endl;

			delete[] log;
		}
		printf("Error: Failed to build program executable!\n");
		exit(1);
	}

	return;

}
