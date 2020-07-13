/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef DEFAULT_PARAMETERS_H_
#define DEFAULT_PARAMETERS_H_

#ifdef SYCL
#include <sycl_vec_maths.hpp>
#else
#include <vector_types.h>
#include <cutil_math.h>
#endif // SYCL
#include <vector>
#include <sstream>
#include <getopt.h>

#include <constant_parameters.h>

extern int optind;

////////////////////////// RUNTIME PARAMETERS //////////////////////

#define DEFAULT_ITERATION_COUNT 3
static const int default_iterations[DEFAULT_ITERATION_COUNT] = { 10, 5, 4 };

const float default_mu = 0.1f;
const bool default_blocking_read = false;
const int default_fps = 0;
const float default_icp_threshold = 1e-5;
const int default_compute_size_ratio = 1;
const int default_integration_rate = 2;
const int default_rendering_rate = 4;
const int default_tracking_rate = 1;
#ifdef SYCL
// Can't access cl::sycl::float3/uint3 element addresses (for Qt GUI callbacks)
const cl_uint3  default_volume_resolution{256, 256, 256};
const cl_float3 default_volume_size{2.f, 2.f, 2.f};
#else
const uint3     default_volume_resolution{256, 256, 256};
const float3    default_volume_size{2.f, 2.f, 2.f};
#endif
const float3 default_initial_pos_factor = make_float3(0.5f, 0.5f, 0.0f);
const bool default_no_gui = false;
const bool default_render_volume_fullsize = false;
const std::string default_dump_volume_file = "";
const std::string default_input_file = "";
const std::string default_log_file = "";

inline std::string pyramid2str(std::vector<int> v) {
	std::ostringstream ss;
	for (std::vector<int>::iterator it = v.begin(); it != v.end(); it++)
		ss << *it << " ";
	return ss.str();

}

static std::string short_options = "qc:d:f:i:l:m:k:o:p:r:s:t:v:y:z:";

static struct option long_options[] =
  {
		    {"compute-size-ratio",     required_argument, 0, 'c'},
		    {"dump-volume",  		   required_argument, 0, 'd'},
		    {"fps",  				   required_argument, 0, 'f'},
		    {"input-file",  		   required_argument, 0, 'i'},
		    {"camera",  			   required_argument, 0, 'k'},
		    {"icp-threshold", 	 	   required_argument, 0, 'l'},
		    {"log-file",  			   required_argument, 0, 'o'},
		    {"mu", 			 		   required_argument, 0, 'm'},
		    {"init-pose",  			   required_argument, 0, 'p'},
		    {"no-gui",  			   no_argument,       0, 'q'},
		    {"integration-rate",  	   required_argument, 0, 'r'},
		    {"volume-size",  		   required_argument, 0, 's'},
		    {"tracking-rate", 		   required_argument, 0, 't'},
		    {"volume-resolution",      required_argument, 0, 'v'},
		    {"pyramid-levels", 		   required_argument, 0, 'y'},
		    {"rendering-rate", required_argument, 0, 'z'},
		    {0, 0, 0, 0}

};

struct Configuration {

	// Possible Parameters

	int compute_size_ratio;
	int integration_rate;
	int rendering_rate;
	int tracking_rate;
#ifdef SYCL
	cl_uint3 volume_resolution; // cl_uint3  can access element addresses for GUI
	cl_float3 volume_size;      // cl_float3        ""
#else
	uint3 volume_resolution;
	float3 volume_size;
#endif
	float3 initial_pos_factor;
	std::vector<int> pyramid;
	std::string dump_volume_file;
	std::string input_file;
	std::string log_file;
	std::ofstream log_filestream;
	std::ostream *log_stream;

	float4 camera;
	bool camera_overrided;

	float mu;
	int fps;
	bool blocking_read;
	float icp_threshold;
	bool no_gui;
	bool render_volume_fullsize;
	inline
	void print_arguments() {
		std ::cerr << "-c  (--compute-size-ratio)       : default is " << default_compute_size_ratio << "   (same size)      " << std::endl;
		std ::cerr << "-d  (--dump-volume) <filename>   : Output volume file              " << std::endl;
		std ::cerr << "-f  (--fps)                      : default is " << default_fps       << std::endl;
		std ::cerr << "-i  (--input-file) <filename>    : Input camera file               " << std::endl;
		std ::cerr << "-k  (--camera)                   : default is defined by input     " << std::endl;
		std ::cerr << "-l  (--icp-threshold)                : default is " << default_icp_threshold << std::endl;
		std ::cerr << "-o  (--log-file) <filename>      : default is stdout               " << std::endl;
		std ::cerr << "-m  (--mu)                       : default is " << default_mu << "               " << std::endl;
#ifdef SYCL
    std ::cerr << "-p  (--init-pose)                : default is " << default_initial_pos_factor.x() << "," << default_initial_pos_factor.y() << "," << default_initial_pos_factor.z() << "     " << std::endl;
#else
		std ::cerr << "-p  (--init-pose)                : default is " << default_initial_pos_factor.x << "," << default_initial_pos_factor.y << "," << default_initial_pos_factor.z << "     " << std::endl;
#endif
		std ::cerr << "-q  (--no-gui)                   : default is to display gui"<<std::endl;
		std ::cerr << "-r  (--integration-rate)         : default is " << default_integration_rate << "     " << std::endl;
#ifdef SYCL
		std ::cerr << "-s  (--volume-size)              : default is " << default_volume_size.s[0] << "," << default_volume_size.s[1] << "," << default_volume_size.s[2] << "      " << std::endl;
#else
		std ::cerr << "-s  (--volume-size)              : default is " << default_volume_size.x << "," << default_volume_size.y << "," << default_volume_size.z << "      " << std::endl;
#endif
		std ::cerr << "-t  (--tracking-rate)            : default is " << default_tracking_rate << "     " << std::endl;
#ifdef SYCL
		std ::cerr << "-v  (--volume-resolution)        : default is " << default_volume_resolution.s[0] << "," << default_volume_resolution.s[1] << "," << default_volume_resolution.s[2] << "    " << std::endl;
#else
		std ::cerr << "-v  (--volume-resolution)        : default is " << default_volume_resolution.x << "," << default_volume_resolution.y << "," << default_volume_resolution.z << "    " << std::endl;
#endif
		std ::cerr << "-y  (--pyramid-levels)           : default is 10,5,4     " << std::endl;
		std ::cerr << "-z  (--rendering-rate)   : default is " << default_rendering_rate << std::endl;
	}
  void print_values(std::ostream& out) {
    time_t rawtime;
		struct tm *timeinfo;
		char buffer[80];
		time(&rawtime);
		timeinfo=localtime(&rawtime);
		strftime(buffer,80,"%Y-%m-%d %I:%M:%S",timeinfo);
		out << "SLAMBench Report run started:\t" << buffer << std::endl<< std::endl;
		out << "Scene properties:" << std::endl<<"=================" << std::endl<< std::endl;
		out << "input-file: " << input_file <<std::endl;
#ifdef SYCL
		out << "volume-size: " << volume_size.s[0] << "," << volume_size.s[1] << "," << volume_size.s[2] << std::endl;		
		out << "camera: "<< (float)camera.x()<<","<< (float)camera.y()<<","<< (float)camera.z()<<","<< (float)camera.w()<<  std::endl;
		out << "init-pose: " << (float)initial_pos_factor.x() << "," << (float)initial_pos_factor.y() << "," << (float)initial_pos_factor.z() << std::endl;
#else
		out << "volume-size: " << volume_size.x << "," << volume_size.y << "," << volume_size.z << std::endl;	
		out << "camera: "<< camera.x<<","<< camera.y<<","<< camera.z<<","<< camera.w<<  std::endl;
		out << "init-pose: " << initial_pos_factor.x << "," << initial_pos_factor.y << "," <<initial_pos_factor.z << std::endl;
#endif		
		out << std::endl;	
		out << "Algorithmic properties:"<<std::endl<<"======================="<<std::endl << std::endl;
		out << "compute-size-ratio: " << compute_size_ratio << std::endl;	

#ifdef SYCL		
		out << "volume-resolution: " << volume_resolution.s[0] << "," << volume_resolution.s[1] << "," << volume_resolution.s[2] << "    " << std::endl;
#else
		out << "volume-resolution: " << volume_resolution.x << "," << volume_resolution.y << "," << volume_resolution.z << "    " << std::endl;		
#endif

		out << "mu: " << mu << std::endl;
		out << "icp-threshold: " << icp_threshold << std::endl;
		out << "pyramid-levels: " ;
		for(int i=0; i< pyramid.size();i++){
			if(i!=0)
				out<<",";
			out<<pyramid[i];
		}		
		out << std::endl;		
		out << "tracking-rate: "  << tracking_rate << std::endl;		
		out << "integration-rate: " << integration_rate << std::endl;		
		out << "rendering-rate: " << rendering_rate << std::endl;
		out << "fps: " << fps << std::endl;
  }

#ifdef SYCL
	inline float3 atof3(char * optarg) {
		float3 res;
		std::istringstream dotargs(optarg);
		std::string s;
		if (getline(dotargs, s, ',')) {
			res.x() = atof(s.c_str());
		} else
			return res;
		if (getline(dotargs, s, ',')) {
			res.y() = atof(s.c_str());
		} else {
			res.y() = res.x();
			res.z() = res.y();
			return res;
		}
		if (getline(dotargs, s, ',')) {
			res.z() = atof(s.c_str());
		} else {
			res.z() = res.y();
		}
		return res;
	}

	inline uint3 atoi3(char * optarg) {
		uint3 res;
		std::istringstream dotargs(optarg);
		std::string s;
		if (getline(dotargs, s, ',')) {
			res.x() = atoi(s.c_str());
		} else
			return res;
		if (getline(dotargs, s, ',')) {
			res.y() = atoi(s.c_str());
		} else {
			res.y() = res.x();
			res.z() = res.y();
			return res;
		}
		if (getline(dotargs, s, ',')) {
			res.z() = atoi(s.c_str());
		} else {
			res.z() = res.y();
		}
		return res;
	}

	inline float4 atof4(char * optarg) {
		float4 res;
		std::istringstream dotargs(optarg);
		std::string s;
		if (getline(dotargs, s, ',')) {
			res.x() = atof(s.c_str());
		} else
			return res;
		if (getline(dotargs, s, ',')) {
			res.y() = atof(s.c_str());
		} else {
			res.y() = res.x();
			res.z() = res.y();
			res.w() = res.z();
			return res;
		}
		if (getline(dotargs, s, ',')) {
			res.z() = atof(s.c_str());
		} else {
			res.z() = res.y();
			res.w() = res.z();
			return res;
		}
		if (getline(dotargs, s, ',')) {
			res.w() = atof(s.c_str());
		} else {
			res.w() = res.z();
		}
		return res;
	}
#else
	inline float3 atof3(char * optarg) {
		float3 res;
		std::istringstream dotargs(optarg);
		std::string s;
		if (getline(dotargs, s, ',')) {
			res.x = atof(s.c_str());
		} else
			return res;
		if (getline(dotargs, s, ',')) {
			res.y = atof(s.c_str());
		} else {
			res.y = res.x;
			res.z = res.y;
			return res;
		}
		if (getline(dotargs, s, ',')) {
			res.z = atof(s.c_str());
		} else {
			res.z = res.y;
		}
		return res;
	}

	inline uint3 atoi3(char * optarg) {
		uint3 res;
		std::istringstream dotargs(optarg);
		std::string s;
		if (getline(dotargs, s, ',')) {
			res.x = atoi(s.c_str());
		} else
			return res;
		if (getline(dotargs, s, ',')) {
			res.y = atoi(s.c_str());
		} else {
			res.y = res.x;
			res.z = res.y;
			return res;
		}
		if (getline(dotargs, s, ',')) {
			res.z = atoi(s.c_str());
		} else {
			res.z = res.y;
		}
		return res;
	}

	inline float4 atof4(char * optarg) {
		float4 res;
		std::istringstream dotargs(optarg);
		std::string s;
		if (getline(dotargs, s, ',')) {
			res.x = atof(s.c_str());
		} else
			return res;
		if (getline(dotargs, s, ',')) {
			res.y = atof(s.c_str());
		} else {
			res.y = res.x;
			res.z = res.y;
			res.w = res.z;
			return res;
		}
		if (getline(dotargs, s, ',')) {
			res.z = atof(s.c_str());
		} else {
			res.z = res.y;
			res.w = res.z;
			return res;
		}
		if (getline(dotargs, s, ',')) {
			res.w = atof(s.c_str());
		} else {
			res.w = res.z;
		}
		return res;
	}
#endif // SYCL

	Configuration(unsigned int argc, char ** argv) {

		compute_size_ratio = default_compute_size_ratio;
		integration_rate = default_integration_rate;
		tracking_rate = default_tracking_rate;
		rendering_rate = default_rendering_rate;
		volume_resolution = default_volume_resolution;
		volume_size = default_volume_size;
		initial_pos_factor = default_initial_pos_factor;

		dump_volume_file = default_dump_volume_file;
		input_file = default_input_file;
		log_file = default_log_file;

		mu = default_mu;
		fps = default_fps;
		blocking_read = default_blocking_read;
		icp_threshold = default_icp_threshold;
		no_gui = default_no_gui;
		render_volume_fullsize = default_render_volume_fullsize;
		camera_overrided = false;

		this->pyramid.clear();
		for (int i = 0; i < DEFAULT_ITERATION_COUNT; i++) {
			this->pyramid.push_back(default_iterations[i]);
		}

		int c;
		int option_index = 0;
		int flagErr = 0;
		optind = 1;
		while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
				&option_index)) != -1)
			switch (c) {
			case 'b':
				this->blocking_read = true;
				std::cerr << "activate blocking read" << std::endl;
				break;
			case 'c':  //   -c  (--compute-size-ratio)
				this->compute_size_ratio = atoi(optarg);
				std::cerr << "update compute_size_ratio to "
						<< this->compute_size_ratio << std::endl;
				if ((this->compute_size_ratio != 1)
						&& (this->compute_size_ratio != 2)
						&& (this->compute_size_ratio != 4)
						&& (this->compute_size_ratio != 8)) {
					std::cerr
							<< "ERROR: --compute-size-ratio (-c) must be 1, 2 ,4 or 8  (was "
							<< optarg << ")\n";
					flagErr++;
				}
				break;
			case 'd':
				this->dump_volume_file = optarg;
				std::cerr << "update dump_volume_file to "
						<< this->dump_volume_file << std::endl;
				break;
			case 'f':  //   -f  (--fps)
				this->fps = atoi(optarg);
				std::cerr << "update fps to " << this->fps << std::endl;

				if (this->fps < 0) {
					std::cerr << "ERROR: --fps (-f) must be >= 0 (was "
							<< optarg << ")\n";
					flagErr++;
				}
				break;
			case 'i':    //   -i  (--input-file)
				this->input_file = optarg;
				std::cerr << "update input_file to " << this->input_file
						<< std::endl;
				struct stat st;
				if (stat(this->input_file.c_str(), &st) != 0) {
					std::cerr << "ERROR: --input-file (-i) does not exist (was "
							<< this->input_file << ")\n";
					flagErr++;
				}
				break;
			case 'k':    //   -k  (--camera)
				this->camera = atof4(optarg);
				this->camera_overrided = true;
#ifdef SYCL
				std::cerr << "update camera to " << (float)this->camera.x() << ","
						<< (float)this->camera.y() << "," << (float)this->camera.z() << ","
						<< (float)this->camera.w() << std::endl;
#else
				std::cerr << "update camera to " << this->camera.x << ","
						<< this->camera.y << "," << this->camera.z << ","
						<< this->camera.w << std::endl;
#endif // SYCL
				break;
			case 'o':    //   -o  (--log-file)
				this->log_file = optarg;
				std::cerr << "update log_file to " << this->log_file
						<< std::endl;
				break;
			case 'l':  //   -l (--icp-threshold)
				this->icp_threshold = atof(optarg);
				std::cerr << "update icp_threshold to " << this->icp_threshold
						<< std::endl;
				break;
			case 'm':   // -m  (--mu)
				this->mu = atof(optarg);
				std::cerr << "update mu to " << this->mu << std::endl;
				break;
			case 'p':    //   -p  (--init-pose)
				this->initial_pos_factor = atof3(optarg);
#ifdef SYCL
				std::cerr << "update init_poseFactors to "
						<< (float)this->initial_pos_factor.x() << ","
						<< (float)this->initial_pos_factor.y() << ","
						<< (float)this->initial_pos_factor.z() << std::endl;
#else
				std::cerr << "update init_poseFactors to "
						<< this->initial_pos_factor.x << ","
						<< this->initial_pos_factor.y << ","
						<< this->initial_pos_factor.z << std::endl;
#endif // SYCL
				break;
			case 'q':
				this->no_gui = true;
				break;
			case 'r':    //   -r  (--integration-rate)
				this->integration_rate = atoi(optarg);
				std::cerr << "update integration_rate to "
						<< this->integration_rate << std::endl;
				if (this->integration_rate < 1) {
					std::cerr
							<< "ERROR: --integration-rate (-r) must >= 1 (was "
							<< optarg << ")\n";
					flagErr++;
				}
				break;
			case 's':    //   -s  (--map-size)
#ifdef SYCL
				this->volume_size = to_cl_float3(atof3(optarg));
				std::cerr << "update map_size to " << this->volume_size.s[0]
						<< "mx" << this->volume_size.s[1] << "mx"
						<< this->volume_size.s[2] << "m" << std::endl;
				if ((this->volume_size.s[0] <= 0.0f) || (this->volume_size.s[1] <= 0.0f)
						|| (this->volume_size.s[2] <= 0.0f)) {
#else
				this->volume_size = atof3(optarg);
				std::cerr << "update map_size to " << this->volume_size.x
						<< "mx" << this->volume_size.y << "mx"
						<< this->volume_size.z << "m" << std::endl;
				if ((this->volume_size.x <= 0) || (this->volume_size.y <= 0)
						|| (this->volume_size.z <= 0)) {
#endif
					std::cerr
							<< "ERROR: --volume-size (-s) all dimensions must > 0 (was "
							<< optarg << ")\n";
					flagErr++;
				}
				break;
			case 't':    //   -t  (--tracking-rate)
				this->tracking_rate = atof(optarg);
				std::cerr << "update tracking_rate to " << this->tracking_rate
						<< std::endl;
				break;
			case 'z':    //   -z  (--rendering-rate)
				this->rendering_rate = atof(optarg);
				std::cerr << "update rendering_rate to " << this->rendering_rate
						<< std::endl;
				break;
			case 'v':    //   -v  (--volumetric-size)
#ifdef SYCL
				this->volume_resolution = to_cl_uint3(atoi3(optarg));
				std::cerr << "update volumetric_size to "
						<< this->volume_resolution.s[0] << "x"
						<< this->volume_resolution.s[1] << "x"
						<< this->volume_resolution.s[2] << std::endl;
				if ((this->volume_resolution.s[0] <= 0)
						|| (this->volume_resolution.s[1] <= 0)
						|| (this->volume_resolution.s[2] <= 0)) {
#else
				this->volume_resolution = atoi3(optarg);
				std::cerr << "update volumetric_size to "
						<< this->volume_resolution.x << "x"
						<< this->volume_resolution.y << "x"
						<< this->volume_resolution.z << std::endl;
				if ((this->volume_resolution.x <= 0)
						|| (this->volume_resolution.y <= 0)
						|| (this->volume_resolution.z <= 0)) {
#endif
					std::cerr
							<< "ERROR: --volume-size (-s) all dimensions must > 0 (was "
							<< optarg << ")\n";
					flagErr++;
				}

				break;
			case 'y': {
				std::istringstream dotargs(optarg);
				std::string s;
				pyramid.clear();
				while (getline(dotargs, s, ',')) {
					pyramid.push_back(atof(s.c_str()));
				}
			}
				std::cerr << "update pyramid levels to " << pyramid2str(pyramid)
						<< std::endl;
				break;
			case 0:
			case '?':
				std::cerr << "Unknown option character -" << char(optopt)
						<< " or bad usage.\n";
				print_arguments();
				exit(0);
			default:
				std::cerr << "GetOpt abort.";
				flagErr = true;
			}

		if (flagErr) {
			std::cerr << "Exited due to " << flagErr << " error"
					<< (flagErr == 1 ? "" : "s")
					<< " in command line options\n";
			exit(1);
		}

	}

};

#endif /* DEFAULT_PARAMETERS_H_ */
