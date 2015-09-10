/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */
#include <SYCL/sycl.hpp>
#include <dagr/dagr.hpp>
#include <kernels.h>

#ifdef __APPLE__
#include <mach/clock.h>
#include <mach/mach.h>

	
	#define TICK()    {if (print_kernel_timing) {\
		host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);\
		clock_get_time(cclock, &tick_clockData);\
		mach_port_deallocate(mach_task_self(), cclock);\
		}}

	#define TOCK(str,size)  {if (print_kernel_timing) {\
		host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);\
		clock_get_time(cclock, &tock_clockData);\
		mach_port_deallocate(mach_task_self(), cclock);\
		std::cerr<< str << " ";\
		if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
		std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;}}
#else
	
	#define TICK()    {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tick_clockData);}}

	#define TOCK(str,size)  {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tock_clockData); std::cerr<< str << " ";\
		if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
		std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;}}

#endif

// input once
float * gaussian;

// inter-frame
#ifdef SYCL
Volume<short2 *> volume;
#else
Volume           volume;
#endif
float3 * vertex;
float3 * normal;

// intra-frame
TrackData * trackingResult;
float* reductionoutput;
float ** ScaledDepth;
float * floatDepth;
Matrix4 oldPose;
Matrix4 raycastPose;
float3 ** inputVertex;
float3 ** inputNormal;

#ifdef SYCL
// sycl specific
cl::sycl::queue q(cl::sycl::intel_selector{});
static_assert(std::is_standard_layout<TrackData>::value,"");
// needed? Depends on depth buffer location/lifetime
// uint2 computationSizeBkp = make_uint2(0, 0);
buffer<float3,1>  *ocl_vertex         = NULL;
buffer<float3,1>  *ocl_normal         = NULL;
buffer<short2,1>  *ocl_volume_data    = NULL;

buffer<float,1>             *ocl_reduce_output_buffer = NULL;
buffer<TrackData,1>         *ocl_trackingResult       = NULL;
buffer<float,1>             *ocl_FloatDepth           = NULL;
buffer<float,1>            **ocl_ScaledDepth          = NULL;
buffer<float3,1>           **ocl_inputVertex          = NULL;
buffer<float3,1>           **ocl_inputNormal          = NULL;
//buffer<ushort,1> *ocl_depth_buffer = NULL; // cl_mem ocl_depth_buffer
/////////// todo 

// cl_mem ocl_output_render_buffer = NULL; // Common buffer for rendering track, depth and volume
buffer<uchar4,1> *ocl_output_render_buffer;
float *reduceOutputBuffer = NULL;

// reduction parameters
static const size_t size_of_group = 64;
static const size_t number_of_groups = 8;

uint2 outputImageSizeBkp = make_uint2(0, 0);
#endif

bool print_kernel_timing = false;
#ifdef __APPLE__
	clock_serv_t cclock;
	mach_timespec_t tick_clockData;
	mach_timespec_t tock_clockData;
#else
	struct timespec tick_clockData;
	struct timespec tock_clockData;
#endif

#ifdef SYCL
struct initVolumeKernelOk {

template <typename T>
static void kernel(item<3> ix, T *data)
{
  uint x = ix[0]; uint y = ix[1]; uint z = ix[2];
  uint3 size{ix.get_range()[0], ix.get_range()[1], ix.get_range()[2]};
  float2 d{1.0f,0.0f};

  data[x + y * size.x() + z * size.x() * size.y()] =
    short2{d.x() * 32766.0f, d.y()};
}

}; // struct
#endif  // SYCL
	
void Kfusion::languageSpecificConstructor() {

	if (getenv("KERNEL_TIMINGS"))
		print_kernel_timing = true;

#ifdef SYCL
  const auto csize = computationSize.x() * computationSize.y();
#else
  const auto csize = computationSize.x   * computationSize.y;
#endif

#ifdef SYCL
  using f_buf  = buffer<float,1>;
  using f3_buf = buffer<float3,1>;
	ocl_FloatDepth = new f_buf(range<1>{csize});

  ocl_ScaledDepth = (f_buf**)  malloc(sizeof(f_buf*)  * iterations.size());
  ocl_inputVertex = (f3_buf**) malloc(sizeof(f3_buf*) * iterations.size());
  ocl_inputNormal = (f3_buf**) malloc(sizeof(f3_buf*) * iterations.size());

  for (unsigned int i = 0; i < iterations.size(); ++i) {
		ocl_ScaledDepth[i] = new f_buf (range<1>{csize/(int)pow(2,i)});
		ocl_inputVertex[i] = new f3_buf(range<1>{csize/(int)pow(2,i)});
		ocl_inputNormal[i] = new f3_buf(range<1>{csize/(int)pow(2,i)});
  }

  ocl_vertex = new f3_buf(range<1>{csize});
  ocl_normal = new f3_buf(range<1>{csize});
  ocl_trackingResult=new buffer<TrackData,1>(range<1>{csize});

  // number_of_groups is 8
	reduceOutputBuffer = (float*) malloc(number_of_groups * 32 * sizeof(float));
  ocl_reduce_output_buffer =
    new f_buf(reduceOutputBuffer, range<1>{32 * number_of_groups});
#endif

	// internal buffers to initialize
	reductionoutput = (float*) calloc(sizeof(float) * 8 * 32, 1);

	ScaledDepth = (float**)  calloc(sizeof(float*)  * iterations.size(), 1);
	inputVertex = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);
	inputNormal = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);

	for (unsigned int i = 0; i < iterations.size(); ++i) {
		ScaledDepth[i] = (float*)  calloc(sizeof(float) *csize / (int) pow(2,i), 1);
		inputVertex[i] = (float3*) calloc(sizeof(float3)*csize / (int) pow(2,i), 1);
		inputNormal[i] = (float3*) calloc(sizeof(float3)*csize / (int) pow(2,i), 1);
	}

	floatDepth     = (float*)     calloc(sizeof(float)     * csize, 1);
	vertex         = (float3*)    calloc(sizeof(float3)    * csize, 1);
	normal         = (float3*)    calloc(sizeof(float3)    * csize, 1);
	trackingResult = (TrackData*) calloc(sizeof(TrackData) * csize, 1);

	// ********* BEGIN : Generate the gaussian *************
	size_t gaussianS = radius * 2 + 1;
	gaussian = (float*) calloc(gaussianS * sizeof(float), 1);
	int x;
	for (unsigned int i = 0; i < gaussianS; i++) {
		x = i - 2;
		gaussian[i] = expf(-(x * x) / (2 * delta * delta));
	}
	// ********* END : Generate the gaussian *************

#ifdef SYCL
  const auto vsize = volumeResolution.x() *
                     volumeResolution.y() * volumeResolution.z();
  ocl_volume_data = new buffer<short2>(range<1>{vsize});

#endif
	volume.init(volumeResolution, volumeDimensions);
	reset();
}

Kfusion::~Kfusion() {

#ifdef SYCL
	if (reduceOutputBuffer)
		free(reduceOutputBuffer);
	reduceOutputBuffer = NULL;

	if (ocl_FloatDepth) {
		delete ocl_FloatDepth;
	  ocl_FloatDepth = NULL;
  }

	for (unsigned int i = 0; i < iterations.size(); ++i) {
		if (ocl_ScaledDepth[i]) {
			delete ocl_ScaledDepth[i];
      ocl_ScaledDepth[i] = NULL;
    }
		if (ocl_inputVertex[i]) {
			delete ocl_inputVertex[i];
      ocl_inputVertex[i] = NULL;
    }
		if (ocl_inputNormal[i]) {
			delete ocl_inputNormal[i];
      ocl_inputNormal[i] = NULL;
    }
/*		if (ocl_inputVertex[i])
			clReleaseMemObject(ocl_inputVertex[i]);
		ocl_inputVertex[i] = NULL;
		if (ocl_inputNormal[i])
			clReleaseMemObject(ocl_inputNormal[i]);
		ocl_inputNormal[i] = NULL; */
	}
	if (ocl_ScaledDepth) {
    free(ocl_ScaledDepth);
    ocl_ScaledDepth = NULL;
  }
	if (ocl_vertex) {
		delete ocl_vertex;
		ocl_vertex = NULL;
  }
	if (ocl_normal) {
		delete ocl_normal;
		ocl_normal = NULL;
  }
	if (ocl_volume_data) {
		delete ocl_volume_data;
		ocl_volume_data = NULL;
  }
	if (ocl_reduce_output_buffer) {
	  delete ocl_reduce_output_buffer;
	  ocl_reduce_output_buffer = NULL;
  }
	if (ocl_trackingResult) {
		delete ocl_trackingResult;
		ocl_trackingResult = NULL;
  }
#endif

	free(reductionoutput);
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		free(ScaledDepth[i]);
		free(inputVertex[i]);
		free(inputNormal[i]);
	}
	free(ScaledDepth);
	free(inputVertex);
	free(inputNormal);

	free(vertex);
	free(normal);
	free(gaussian);

	volume.release();
}
void Kfusion::reset() {
	initVolumeKernel(volume);
}
void init() {
}
;
// stub
void clean() {
}
;
// stub

#ifdef SYCL
template <typename T>
void initVolumeKernel(Volume<T> volume) {
	TICK();
	for (unsigned int x = 0; x < volume.size.x(); x++)
		for (unsigned int y = 0; y < volume.size.y(); y++) {
			for (unsigned int z = 0; z < volume.size.z(); z++) {
				//std::cout <<  x << " " << y << " " << z <<"\n";
        /*const*/ float2 w{1.0f, 0.0f};
				volume.setints(x, y, z, w /*make_float2(1.0f, 0.0f)*/); // w = nonconst
			}
		}
	TOCK("initVolumeKernel", volume.size.x() * volume.size.y() * volume.size.z());
}
#else
void initVolumeKernel(Volume volume) {
	TICK();
	for (unsigned int x = 0; x < volume.size.x; x++)
		for (unsigned int y = 0; y < volume.size.y; y++) {
			for (unsigned int z = 0; z < volume.size.z; z++) {
				//std::cout <<  x << " " << y << " " << z <<"\n";
        /*const*/ float2 w{1.0f, 0.0f};
				volume.setints(x, y, z, w /*make_float2(1.0f, 0.0f)*/); // w = nonconst
			}
		}
	TOCK("initVolumeKernel", volume.size.x * volume.size.y * volume.size.z);
}
#endif

void bilateralFilterKernel(float* out, const float* in, uint2 size,
		const float * gaussian, float e_d, int r) {
	TICK()
		uint y;
		float e_d_squared_2 = e_d * e_d * 2;
#ifdef SYCL
	const uint size_x = size.x(); const uint size_y = size.y();
#else
	const uint size_x = size.x;   const uint size_y = size.y;
#endif

#pragma omp parallel for \
	    shared(out),private(y)   
		for (y = 0; y < size_y; y++) {
			for (uint x = 0; x < size_x; x++) {
				uint pos = x + y * size_x;
				if (in[pos] == 0) {
					out[pos] = 0;
					continue;
				}

				float sum = 0.0f;
				float t = 0.0f;

				const float center = in[pos];

				for (int i = -r; i <= r; ++i) {
					for (int j = -r; j <= r; ++j) {
						uint2 curPos = make_uint2(clamp(x + i, 0u, size_x - 1),
                                      clamp(y + j, 0u, size_y - 1));
#ifdef SYCL
						const float curPix = in[curPos.x() + curPos.y() * size_x];
#else
						const float curPix = in[curPos.x   + curPos.y   * size_x];
#endif
						if (curPix > 0) {
							const float mod = sq(curPix - center);
							const float factor = gaussian[i + r]
									* gaussian[j + r]
									* expf(-mod / e_d_squared_2);
							t += factor * curPix;
							sum += factor;
						}
					}
				}
				out[pos] = t / sum;
			}
		}
		TOCK("bilateralFilterKernel", size_x * size_y);
}

#ifdef SYCL
template <typename F3>
inline F3 myrotate(/*const*/ Matrix4 M, const F3 v) {
	return F3{my_dot(F3{M.data[0].x(), M.data[0].y(), M.data[0].z()}, v),
            my_dot(F3{M.data[1].x(), M.data[1].y(), M.data[1].z()}, v),
            my_dot(F3{M.data[2].x(), M.data[2].y(), M.data[2].z()}, v)};
}

template <typename F3>
inline F3 Mat4TimeFloat3(/*const*/ Matrix4 M, const F3 v) {
	return
  F3{cl::sycl::dot(F3{M.data[0].x(), M.data[0].y(), M.data[0].z()}, v) + M.data[0].w(),
     cl::sycl::dot(F3{M.data[1].x(), M.data[1].y(), M.data[1].z()}, v) + M.data[1].w(),
     cl::sycl::dot(F3{M.data[2].x(), M.data[2].y(), M.data[2].z()}, v) + M.data[2].w()};
}

template <typename T>
inline void setVolume(Volume<T> v, uint3 pos, float2 d) {
	v.data[pos.x() +
         pos.y() * v.size.x() +
         pos.z() * v.size.x() * v.size.y()] = short2{d.x() * 32766.0f, d.y()};
}

template <typename T>
inline float3 posVolume(/*const*/ Volume<T> v, /*const*/ uint3 p) {
	return float3{(p.x() + 0.5f) * v.dim.x() / v.size.x(),
                (p.y() + 0.5f) * v.dim.y() / v.size.y(),
                (p.z() + 0.5f) * v.dim.z() / v.size.z()};
}

template <typename T>
inline float2 getVolume(/*const*/ Volume<T> v, /*const*/ uint3 pos) {
  /*const*/ short2 d = v.data[pos.x() +   // Making d a ref fixes it.
                              pos.y() * v.size.x() +
                              pos.z() * v.size.x() * v.size.y()];
	return float2{1,2};//d.x() * 0.00003051944088f, d.y()}; //  / 32766.0f
}
#endif

#ifdef SYCL

struct depth2vertexKernel {

// vertex is actually an array of float3 (T == float3).
// vertexSize and depthSize are unused.
template <typename T, typename U>
static void kernel(item<2> ix, T *vertex, const uint2 vertexSize,
                   const U *depth, const uint2 depthSize, const Matrix4 invK)
{
  int2   pixel{ix[0],ix[1]};
  float3 vert{ix[0],ix[1],1.0f};
  float3 res{0,0,0};

  float elem = depth[pixel.x() + ix.get_range()[0] * pixel.y()];
  if (elem > 0) {
    float3 tmp3{pixel.x(), pixel.y(), 1.f};
//          res = elem * myrotate(invK, tmp3); // SYCL needs this (*) operator
    float3 rot = myrotate(invK, tmp3);
    res.x() = elem * rot.x();
    res.y() = elem * rot.y();
    res.z() = elem * rot.z();
  }

  // cl::sycl::vstore3(res, pixel.x() + ix.get_range()[0] * pixel.y(),vertex); 	// vertex[pixel] = 
  // This use of 4*32 bits data is fine; but if copied back, ensure data
  // is similarly aligned
  vertex[pixel.x() + ix.get_range()[0] * pixel.y()] = res;
}

}; // struct

#else // SYCL

void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize,
		const Matrix4 invK) {
	TICK();
	unsigned int x, y;
#pragma omp parallel for \
         shared(vertex), private(x, y)
	for (y = 0; y < imageSize.y; y++) {
		for (x = 0; x < imageSize.x; x++) {

			if (depth[x + y * imageSize.x] > 0) {
				vertex[x + y * imageSize.x] = depth[x + y * imageSize.x]
						* (rotate(invK, make_float3(x, y, 1.f)));
			} else {
				vertex[x + y * imageSize.x] = make_float3(0);
			}
		}
	}
	TOCK("depth2vertexKernel", imageSize.x * imageSize.y);
}

#endif // SYCL

#ifdef SYCL

struct vertex2normalKernel {

// normal and vertex are actually arrays of float3
template <typename T>
static void kernel(item<2> ix,       T *normal, const uint2 normalSize,
                               const T *verte_, const uint2 vertexSize)
{
  using const_float3_as1_t = const __attribute__((address_space(1))) float3&;
  static_assert(std::is_same<decltype(verte_[0]),const_float3_as1_t>::value,"");

  // otherwise x=vertex[0] is an error. See const_vec_ptr.cpp
  const float3 *vertex = verte_;
  // auto vertex = verte_; // more generic?
  static_assert(std::is_same<decltype(vertex[0]),const float3&>::value,"");

  uint2  pixel{ix[0],ix[1]};
  uint2  vleft{max((int)(pixel.x())-1,0),                           pixel.y()};
  uint2 vright{min((int)(pixel.x())+1, (int)ix.get_range()[0]-1),   pixel.y()};
  uint2    vup{pixel.x(),                           max((int)(pixel.y())-1,0)};
  uint2  vdown{pixel.x(),   min((int)(pixel.y())+1, (int)ix.get_range()[1]-1)};

  /*const*/ float3 left  = vertex[vleft.x()  + ix.get_range()[0] * vleft.y()];
  /*const*/ float3 right = vertex[vright.x() + ix.get_range()[0] * vright.y()];
  /*const*/ float3 up    = vertex[vup.x()    + ix.get_range()[0] * vup.y()];
  /*const*/ float3 down  = vertex[vdown.x()  + ix.get_range()[0] * vdown.y()];

  if (left.z() == 0 || right.z() == 0|| up.z() == 0 || down.z() == 0) {
    float3 invalid3{INVALID,INVALID,INVALID};
    normal[pixel.x() + ix.get_range()[0] * pixel.y()] = invalid3;
    return;
  }

  const float3 dxv = right - left;
  const float3 dyv = down  - up;
  normal[pixel.x() + ix.get_range()[0] * pixel.y()] = normalize(cross(dyv,dxv));
}

}; // struct

#else // SYCL

void vertex2normalKernel(float3 * out, const float3 * in, uint2 imageSize) {
	TICK();
	unsigned int x, y;
#ifdef SYCL
	const uint imgSize_x = imageSize.x(); const uint imgSize_y = imageSize.y();
#else
	const uint imgSize_x = imageSize.x;   const uint imgSize_y = imageSize.y;
#endif
#pragma omp 0 // parallel for \
        shared(out), private(x,y)
	for (y = 0; y < imgSize_y; y++) {
		for (x = 0; x < imgSize_x; x++) {
			/*const*/ uint2 pleft  = make_uint2(max(int(x)-1,0),                y);
			/*const*/ uint2 pright = make_uint2(min(int(x)+1,(int)imgSize_x-1), y);
			/*const*/ uint2 pup    = make_uint2(x, max(int(y)-1, 0));
			/*const*/ uint2 pdown  = make_uint2(x, min(int(y)+1, ((int)imgSize_y)-1));

#ifdef SYCL
			/*const*/ float3 left  = in[pleft.x()  + imgSize_x * pleft.y()];
			/*const*/ float3 right = in[pright.x() + imgSize_x * pright.y()];
			/*const*/ float3 up    = in[pup.x()    + imgSize_x * pup.y()];
			/*const*/ float3 down  = in[pdown.x()  + imgSize_x * pdown.y()];

			if (left.z() == 0 || right.z() == 0 || up.z() == 0 || down.z() == 0) {
				out[x + y * imgSize_x].x() = INVALID;
				continue;
			}
#else
			const float3 left  = in[pleft.x  + imgSize_x * pleft.y];
			const float3 right = in[pright.x + imgSize_x * pright.y];
			const float3 up    = in[pup.x    + imgSize_x * pup.y];
			const float3 down  = in[pdown.x  + imgSize_x * pdown.y];

			if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
				out[x + y * imgSize_x].x = INVALID;
				continue;
			}
#endif
			const float3 dxv = right - left;
			const float3 dyv = down - up;
			out[x + y * imgSize_x] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
		}
	}
	TOCK("vertex2normalKernel", imgSize_x * imgSize_y);
}

#endif // SYCL

void new_reduce(int blockIndex, float * out, TrackData* J, /*const*/ uint2 Jsize,
		/*const*/ uint2 size) {
	float *sums = out + blockIndex * 32;

	float * jtj = sums + 7;
	float * info = sums + 28;
	for (uint i = 0; i < 32; ++i)
		sums[i] = 0;
	float sums0, sums1, sums2, sums3, sums4, sums5, sums6, sums7, sums8, sums9,
			sums10, sums11, sums12, sums13, sums14, sums15, sums16, sums17,
			sums18, sums19, sums20, sums21, sums22, sums23, sums24, sums25,
			sums26, sums27, sums28, sums29, sums30, sums31;
	sums0 = 0.0f;
	sums1 = 0.0f;
	sums2 = 0.0f;
	sums3 = 0.0f;
	sums4 = 0.0f;
	sums5 = 0.0f;
	sums6 = 0.0f;
	sums7 = 0.0f;
	sums8 = 0.0f;
	sums9 = 0.0f;
	sums10 = 0.0f;
	sums11 = 0.0f;
	sums12 = 0.0f;
	sums13 = 0.0f;
	sums14 = 0.0f;
	sums15 = 0.0f;
	sums16 = 0.0f;
	sums17 = 0.0f;
	sums18 = 0.0f;
	sums19 = 0.0f;
	sums20 = 0.0f;
	sums21 = 0.0f;
	sums22 = 0.0f;
	sums23 = 0.0f;
	sums24 = 0.0f;
	sums25 = 0.0f;
	sums26 = 0.0f;
	sums27 = 0.0f;
	sums28 = 0.0f;
	sums29 = 0.0f;
	sums30 = 0.0f;
	sums31 = 0.0f;
#ifdef SYCL
	const uint size_x  = size.x(); const uint size_y = size.y();
  const uint Jsize_x = Jsize.x();
#else
	const uint size_x  = size.x;   const uint size_y = size.y;
  const uint Jsize_x = Jsize.x;
#endif
// comment me out to try coarse grain parallelism 
#pragma omp parallel for reduction(+:sums0,sums1,sums2,sums3,sums4,sums5,sums6,sums7,sums8,sums9,sums10,sums11,sums12,sums13,sums14,sums15,sums16,sums17,sums18,sums19,sums20,sums21,sums22,sums23,sums24,sums25,sums26,sums27,sums28,sums29,sums30,sums31)
	for (uint y = blockIndex; y < size_y; y += 8) {
		for (uint x = 0; x < size_x; x++) {

			const TrackData & row = J[(x + y * Jsize_x)]; // ...
			if (row.result < 1) {
				// accesses sums[28..31]
				/*(sums+28)[1]*/sums29 += row.result == -4 ? 1 : 0;
				/*(sums+28)[2]*/sums30 += row.result == -5 ? 1 : 0;
				/*(sums+28)[3]*/sums31 += row.result > -4 ? 1 : 0;

				continue;
			}
			// Error part
			/*sums[0]*/sums0 += row.error * row.error;

			// JTe part
			/*for(int i = 0; i < 6; ++i)
			 sums[i+1] += row.error * row.J[i];*/
			sums1 += row.error * row.J[0];
			sums2 += row.error * row.J[1];
			sums3 += row.error * row.J[2];
			sums4 += row.error * row.J[3];
			sums5 += row.error * row.J[4];
			sums6 += row.error * row.J[5];

			// JTJ part, unfortunatly the double loop is not unrolled well...
			/*(sums+7)[0]*/sums7 += row.J[0] * row.J[0];
			/*(sums+7)[1]*/sums8 += row.J[0] * row.J[1];
			/*(sums+7)[2]*/sums9 += row.J[0] * row.J[2];
			/*(sums+7)[3]*/sums10 += row.J[0] * row.J[3];

			/*(sums+7)[4]*/sums11 += row.J[0] * row.J[4];
			/*(sums+7)[5]*/sums12 += row.J[0] * row.J[5];

			/*(sums+7)[6]*/sums13 += row.J[1] * row.J[1];
			/*(sums+7)[7]*/sums14 += row.J[1] * row.J[2];
			/*(sums+7)[8]*/sums15 += row.J[1] * row.J[3];
			/*(sums+7)[9]*/sums16 += row.J[1] * row.J[4];

			/*(sums+7)[10]*/sums17 += row.J[1] * row.J[5];

			/*(sums+7)[11]*/sums18 += row.J[2] * row.J[2];
			/*(sums+7)[12]*/sums19 += row.J[2] * row.J[3];
			/*(sums+7)[13]*/sums20 += row.J[2] * row.J[4];
			/*(sums+7)[14]*/sums21 += row.J[2] * row.J[5];

			/*(sums+7)[15]*/sums22 += row.J[3] * row.J[3];
			/*(sums+7)[16]*/sums23 += row.J[3] * row.J[4];
			/*(sums+7)[17]*/sums24 += row.J[3] * row.J[5];

			/*(sums+7)[18]*/sums25 += row.J[4] * row.J[4];
			/*(sums+7)[19]*/sums26 += row.J[4] * row.J[5];

			/*(sums+7)[20]*/sums27 += row.J[5] * row.J[5];

			// extra info here
			/*(sums+28)[0]*/sums28 += 1;

		}
	}
	sums[0] = sums0;
	sums[1] = sums1;
	sums[2] = sums2;
	sums[3] = sums3;
	sums[4] = sums4;
	sums[5] = sums5;
	sums[6] = sums6;
	sums[7] = sums7;
	sums[8] = sums8;
	sums[9] = sums9;
	sums[10] = sums10;
	sums[11] = sums11;
	sums[12] = sums12;
	sums[13] = sums13;
	sums[14] = sums14;
	sums[15] = sums15;
	sums[16] = sums16;
	sums[17] = sums17;
	sums[18] = sums18;
	sums[19] = sums19;
	sums[20] = sums20;
	sums[21] = sums21;
	sums[22] = sums22;
	sums[23] = sums23;
	sums[24] = sums24;
	sums[25] = sums25;
	sums[26] = sums26;
	sums[27] = sums27;
	sums[28] = sums28;
	sums[29] = sums29;
	sums[30] = sums30;
	sums[31] = sums31;

}
void reduceKernel(float * out, TrackData* J, const uint2 Jsize,
		const uint2 size) {
	TICK();
	int blockIndex;
#ifdef OLDREDUCE
#pragma omp parallel for private (blockIndex)
#endif
	for (blockIndex = 0; blockIndex < 8; blockIndex++) {

#ifdef OLDREDUCE
		float S[112][32]; // this is for the final accumulation
		// we have 112 threads in a blockdim
		// and 8 blocks in a gridDim?
		// ie it was launched as <<<8,112>>>
		uint sline;// threadIndex.x
		float sums[32];

		for(int threadIndex = 0; threadIndex < 112; threadIndex++) {
			sline = threadIndex;
			float * jtj = sums+7;
			float * info = sums+28;
			for(uint i = 0; i < 32; ++i) sums[i] = 0;

			for(uint y = blockIndex; y < size.y(); y += 8 /*gridDim.x()*/) {
				for(uint x = sline; x < size.x; x += 112 /*blockDim.x()*/) {
					const TrackData & row = J[(x + y * Jsize.x())]; // ...

					if(row.result < 1) {
						// accesses S[threadIndex][28..31]
						info[1] += row.result == -4 ? 1 : 0;
						info[2] += row.result == -5 ? 1 : 0;
						info[3] += row.result > -4 ? 1 : 0;
						continue;
					}
					// Error part
					sums[0] += row.error * row.error;

					// JTe part
					for(int i = 0; i < 6; ++i)
					sums[i+1] += row.error * row.J[i];

					// JTJ part, unfortunatly the double loop is not unrolled well...
					jtj[0] += row.J[0] * row.J[0];
					jtj[1] += row.J[0] * row.J[1];
					jtj[2] += row.J[0] * row.J[2];
					jtj[3] += row.J[0] * row.J[3];

					jtj[4] += row.J[0] * row.J[4];
					jtj[5] += row.J[0] * row.J[5];

					jtj[6] += row.J[1] * row.J[1];
					jtj[7] += row.J[1] * row.J[2];
					jtj[8] += row.J[1] * row.J[3];
					jtj[9] += row.J[1] * row.J[4];

					jtj[10] += row.J[1] * row.J[5];

					jtj[11] += row.J[2] * row.J[2];
					jtj[12] += row.J[2] * row.J[3];
					jtj[13] += row.J[2] * row.J[4];
					jtj[14] += row.J[2] * row.J[5];

					jtj[15] += row.J[3] * row.J[3];
					jtj[16] += row.J[3] * row.J[4];
					jtj[17] += row.J[3] * row.J[5];

					jtj[18] += row.J[4] * row.J[4];
					jtj[19] += row.J[4] * row.J[5];

					jtj[20] += row.J[5] * row.J[5];

					// extra info here
					info[0] += 1;
				}
			}

			for(int i = 0; i < 32; ++i) { // copy over to shared memory
				S[sline][i] = sums[i];
			}
			// WE NO LONGER NEED TO DO THIS AS the threads execute sequentially inside a for loop

		} // threads now execute as a for loop.
		  //so the __syncthreads() is irrelevant

		for(int ssline = 0; ssline < 32; ssline++) { // sum up columns and copy to global memory in the final 32 threads
			for(unsigned i = 1; i < 112 /*blockDim.x*/; ++i) {
				S[0][ssline] += S[i][ssline];
			}
			out[ssline+blockIndex*32] = S[0][ssline];
		}
#else 
		new_reduce(blockIndex, out, J, Jsize, size);
#endif

	}

	TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
	for (int j = 1; j < 8; ++j) {
		values[0] += values[j];
		//std::cerr << "REDUCE ";for(int ii = 0; ii < 32;ii++)
		//std::cerr << values[0][ii] << " ";
		//std::cerr << "\n";
	}
	TOCK("reduceKernel", 512);
}

#ifdef SYCL
struct trackKernel {
// inVertex, inNormal, refVertex, and refNormal are really float3 arrays.
template <typename T, typename U>
static void kernel(item<2> ix,
                         T *output,            /*const*/ uint2 outputSize,
                   const U *inVerte_,          const uint2 inVertexSize,
                   const U *inNorma_,          const uint2 inNormalSize,
                   const U *refVertex,         const uint2 refVertexSize,
                   const U *refNorma_,         const uint2 refNormalSize,
                   const Matrix4 Ttrack,       const Matrix4 view,
                   const float dist_threshold, const float normal_threshold)
{
  const float3 *inNormal  = inNorma_;  // See const_vec_ptr.cpp
  const float3 *inVertex  = inVerte_;  // ""
  const float3 *refNormal = refNorma_; // ""
  uint2 pixel{ix[0],ix[1]};
  TrackData &row = output[pixel.x() + outputSize.x() * pixel.y()];

  float3 inNormalPixel = inNormal[pixel.x() + ix.get_range()[0] * pixel.y()];
  if (inNormalPixel.x() == INVALID) {
    row.result = -1;
    return;
  }

  float3 inVertexPixel = inVertex[pixel.x() + ix.get_range()[0] * pixel.y()];
  /*const*/ float3 projectedVertex = Mat4TimeFloat3(Ttrack, inVertexPixel);
  /*const*/ float3 projectedPos    = Mat4TimeFloat3(view, projectedVertex);
  /*const*/ float2 projPixel{projectedPos.x() / projectedPos.z() + 0.5f,
                             projectedPos.y() / projectedPos.z() + 0.5f};
  if (projPixel.x() < 0 || projPixel.x() > outputSize.x()-1 ||
      projPixel.y() < 0 || projPixel.y() > outputSize.y()-1) {
    row.result = -2;
    return;
  }

  /*const*/ uint2 refPixel{projPixel.x(), projPixel.y()};
  /*const*/ float3 referenceNormal =
    refNormal[refPixel.x() + outputSize.x() * refPixel.y()];
  if (referenceNormal.x() == INVALID) {
    row.result = -3;
    return;
  }

  const float3 diff = refVertex[refPixel.x() + outputSize.x() * refPixel.y()] -
                      projectedVertex;
  const float3 projectedNormal = myrotate(Ttrack, inNormalPixel);

  if (length(diff) > dist_threshold) {
    row.result = -4;
    return;  // !!!!!!!!!!!!
  }
  if (dot(projectedNormal,referenceNormal)<normal_threshold) {
    row.result = -5;
    return;
  }

  row.result = 1;
  row.error  = dot(referenceNormal, diff);
  *((float3 *)(row.J + 0)) = referenceNormal; // a la vstore3
  *((float3 *)(row.J + 3)) = cross(projectedVertex, referenceNormal);
  // row.J + 0 -> row.J[0:2]          row.J + 3 ->  row.J[3:5]
}
}; // struct
#else
void trackKernel(TrackData* output, const float3* inVertex,
		const float3* inNormal, uint2 inSize, const float3* refVertex,
		const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
		const Matrix4 view, const float dist_threshold,
		const float normal_threshold) {
	TICK();
	uint2 pixel = make_uint2(0, 0);
	unsigned int pixely, pixelx;
#pragma omp parallel for \
	    shared(output), private(pixel,pixelx,pixely)
	for (pixely = 0; pixely < inSize.y; pixely++) {
		for (pixelx = 0; pixelx < inSize.x; pixelx++) {
			pixel.x = pixelx;
			pixel.y = pixely;

			TrackData & row = output[pixel.x + pixel.y * refSize.x];

			if (inNormal[pixel.x + pixel.y * inSize.x].x == INVALID) {
				row.result = -1;
				continue;
			}

			const float3 projectedVertex = Ttrack
					* inVertex[pixel.x + pixel.y * inSize.x];
			const float3 projectedPos = view * projectedVertex;
			const float2 projPixel = make_float2(
					projectedPos.x / projectedPos.z + 0.5f,
					projectedPos.y / projectedPos.z + 0.5f);
			if (projPixel.x < 0 || projPixel.x > refSize.x - 1
					|| projPixel.y < 0 || projPixel.y > refSize.y - 1) {
				row.result = -2;
				continue;
			}

			const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
			const float3 referenceNormal = refNormal[refPixel.x
					+ refPixel.y * refSize.x];

			if (referenceNormal.x == INVALID) {
				row.result = -3;
				continue;
			}

			const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x]
					- projectedVertex;
			const float3 projectedNormal = rotate(Ttrack,
					inNormal[pixel.x + pixel.y * inSize.x]);

			if (length(diff) > dist_threshold) {
				row.result = -4;
				continue;
			}
			if (dot(projectedNormal, referenceNormal) < normal_threshold) {
				row.result = -5;
				continue;
			}
			row.result = 1;
			row.error = dot(referenceNormal, diff);
			((float3 *) row.J)[0] = referenceNormal;
			((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
		}
	}
	TOCK("trackKernel", inSize.x * inSize.y);
}
#endif // SYCL

void mm2metersKernel(float *out, uint2 outSize, const ushort *in, uint2 inSize)
{
	TICK();
#ifdef SYCL
	const uint inSize_x  = inSize.x();  const uint inSize_y  = inSize.y();
	const uint outSize_x = outSize.x(); const uint outSize_y = outSize.y();
#else
	const uint inSize_x  = inSize.x;  const uint inSize_y  = inSize.y;
	const uint outSize_x = outSize.x; const uint outSize_y = outSize.y;
#endif

	// Check for unsupported conditions
	if ((inSize_x < outSize_x) || (inSize_y < outSize_y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize_x % outSize_x != 0) || (inSize_y % outSize_y != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize_x / outSize_x != inSize_y / outSize_y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = inSize_x / outSize_x;
	unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
	for (y = 0; y < outSize_y; y++)
		for (unsigned int x = 0; x < outSize_x; x++) {
			out[x + outSize_x * y] = in[x * ratio + inSize_x * y * ratio]
					/ 1000.0f;
		}
	TOCK("mm2metersKernel", outSize_x * outSize_y);
}

#ifdef SYCL

struct halfSampleRobustImageKernel {

template <typename T>
static void kernel(item<2> ix, T *out, const T *in,
                   /*const*/ uint2 inSize, const float e_d, const int r)
{
  uint2 pixel{ix[0],ix[1]};
  uint2 outSize{inSize.x() / 2, inSize.y() / 2};

  /*const*/ uint2 centerPixel{2*pixel.x(), 2*pixel.y()};

  float sum = 0.0f;
  float t   = 0.0f;
  const float center = in[centerPixel.x()+centerPixel.y()*inSize.x()];
  for(int i = -r + 1; i <= r; ++i) {
    for(int j = -r + 1; j <= r; ++j) {
      const int2 x{centerPixel.x()+j, centerPixel.y()+i};
      const int2 minval{0,0};
      const int2 maxval{inSize.x()-1, inSize.y()-1};
            int2 from{clamp(x,minval,maxval)};
      float current = in[from.x() + from.y() * inSize.x()];
      if (fabs(current - center) < e_d) {
        sum += 1.0f;
        t += current;
      }
    }
  }
  out[pixel.x() + pixel.y() * outSize.x()] = t / sum;
}

}; // struct

#else // SYCL

void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize,
		const float e_d, const int r) {
	TICK();
	uint2 outSize = make_uint2(inSize.x / 2, inSize.y / 2);
	unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
	for (y = 0; y < outSize.y; y++) {
		for (unsigned int x = 0; x < outSize.x; x++) {
			uint2 pixel = make_uint2(x, y);
			const uint2 centerPixel = 2 * pixel;

			float sum = 0.0f;
			float t = 0.0f;
			const float center = in[centerPixel.x
					+ centerPixel.y * inSize.x];
			for (int i = -r + 1; i <= r; ++i) {
				for (int j = -r + 1; j <= r; ++j) {
					uint2 cur = make_uint2(
							clamp(
									make_int2(centerPixel.x + j,
											centerPixel.y + i), make_int2(0),
									make_int2(2 * outSize.x - 1,
											2 * outSize.y - 1)));
					float current = in[cur.x + cur.y * inSize.x];
					if (fabsf(current - center) < e_d) {
						sum += 1.0f;
						t += current;
					}
				}
			}
			out[pixel.x + pixel.y * outSize.x] = t / sum;
		}
	}
	TOCK("halfSampleRobustImageKernel", outSize.x * outSize.y);
}

#endif // SYCL

#ifdef SYCL

struct integrateKernel {

template <typename I, typename T, typename U>
static void kernel(I ix, T *v_data, const uint3 v_size, const float3 v_dim,
                   const U *depth, /*const*/ uint2 depthSize,
                   const Matrix4 invTrack, const Matrix4 K, const float mu,
                   const float maxweight, const float3 delta,
                   const float3 cameraDelta)
{
  Volume<decltype(&v_data[0])> vol;
  vol.data = &v_data[0]; vol.size = v_size; vol.dim = v_dim;

  uint3 pix{ix[0],ix[1],0};
  const int sizex = ix.get_range()[0];

  float3 pos     = Mat4TimeFloat3(invTrack, posVolume(vol,pix));
  float3 cameraX = Mat4TimeFloat3(K, pos);

  for (pix.z() = 0; pix.z() < vol.size.z();
         pix.z() = pix.z()+1, pos += delta, cameraX += cameraDelta)
  {
    if (pos.z() < 0.0001f) // some near plane constraint
      continue;

    /*const*/ float2 pixel{cameraX.x()/cameraX.z() + 0.5f,
                           cameraX.y()/cameraX.z() + 0.5f};

    if (pixel.x() < 0 || pixel.x() > depthSize.x()-1 ||
        pixel.y() < 0 || pixel.y() > depthSize.y()-1)
      continue;

    /*const*/ uint2 px{pixel.x(), pixel.y()};
    float depthpx = depth[px.x() + depthSize.x() * px.y()];

    if (depthpx == 0)
      continue;

    const float diff = (depthpx - cameraX.z()) *
         cl::sycl::sqrt(1+sq(pos.x()/pos.z()) + sq(pos.y()/pos.z()));

    if (diff > -mu)
    {
      const float sdf = fmin(1.f, diff/mu);
      float2 data = getVolume(vol,pix);
      data.x() = clamp((data.y()*data.x() + sdf)/(data.y() + 1),-1.f,1.f);
      data.y() = fmin(data.y()+1, maxweight);
      setVolume(vol,pix,data);
    }
  }
}
}; // struct 

#else
void integrateKernel(Volume vol, const float* depth, uint2 depthSize,
		const Matrix4 invTrack, const Matrix4 K, const float mu,
		const float maxweight) {
	TICK();
	const float3 delta =
    rotate(invTrack, make_float3(0, 0, vol.dim.z / vol.size.z));
	const float3 cameraDelta = rotate(K, delta);
	unsigned int y;
#pragma omp parallel for \
        shared(vol), private(y)
	for (y = 0; y < vol.size.y; y++)
		for (unsigned int x = 0; x < vol.size.x; x++) {

			uint3 pix = make_uint3(x, y, 0); //pix.x() = x;pix.y() = y;
			float3 pos = invTrack * vol.pos(pix);
			float3 cameraX = K * pos;

			for (pix.z = 0; pix.z < vol.size.z;
					++pix.z,pos += delta,cameraX += cameraDelta) {
				if (pos.z < 0.0001f) // some near plane constraint
					continue;
				const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
						cameraX.y / cameraX.z + 0.5f);
				if (pixel.x < 0 || pixel.x > depthSize.x - 1 || pixel.y < 0
						|| pixel.y > depthSize.y - 1)
					continue;
				const uint2 px = make_uint2(pixel.x, pixel.y);
				if (depth[px.x + px.y * depthSize.x] == 0)
					continue;
				const float diff = (depth[px.x + px.y * depthSize.x] - cameraX.z) *
              std::sqrt(1 + sq(pos.x/pos.z) + sq(pos.y/pos.z));
				if (diff > -mu) {
					const float sdf = fminf(1.f, diff / mu);
					float2 data = vol[pix];
					data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f, 1.f);
					data.y = fminf(data.y + 1, maxweight);
					vol.set(pix, data);
				}
			}
		}
	TOCK("integrateKernel", vol.size.x   * vol.size.y); // * vol.size.z ?
}
#endif

#ifdef SYCL
template <typename T>
float4 raycast(/*const*/ Volume<T> volume,
#else
float4 raycast(const     Volume    volume,
#endif
    /*const*/ uint2 pos, const Matrix4 view, const float nearPlane,
    const float farPlane, const float step, const float largestep) {

	const float3 origin = get_translation(view);
#ifdef SYCL
	const float3 direction = rotate(view, make_float3(pos.x(), pos.y(), 1.f));
#else
	const float3 direction = rotate(view, make_float3(pos.x,   pos.y,   1.f));
#endif

	// intersect ray with a box
	// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
	// compute intersection of ray with all six bbox planes
	const float3 invR = make_float3(1.0f) / direction;
	const float3 tbot = -1 * invR * origin;
	const float3 ttop = invR * (volume.dim - origin);

	// re-order intersections to find smallest and largest on each axis
	/*const*/ float3 tmin = fminf(ttop, tbot);
	/*const*/ float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
#ifdef SYCL
	const float largest_tmin =
    fmaxf(fmaxf(tmin.x(), tmin.y()), fmaxf(tmin.x(), tmin.z()));
	const float smallest_tmax =
    fminf(fminf(tmax.x(), tmax.y()), fminf(tmax.x(), tmax.z()));
#else
	const float largest_tmin  = fmaxf(fmaxf(tmin.x,tmin.y),fmaxf(tmin.x,tmin.z));
	const float smallest_tmax = fminf(fminf(tmax.x,tmax.y),fminf(tmax.x,tmax.z));
#endif

	// check against near and far plane
	const float tnear = fmaxf(largest_tmin, nearPlane);
	const float tfar = fminf(smallest_tmax, farPlane);

	if (tnear < tfar) {
		// first walk with largesteps until we found a hit
		float t = tnear;
		float stepsize = largestep;
#ifdef SYCL
    float3 tmp{origin + direction * t};
		float f_t = volume.interp(tmp); // interp is nonconst
#else
		float f_t = volume.interp(origin + direction * t);
#endif
		float f_tt = 0;
		if (f_t > 0) { // ups, if we were already in it, then don't render anything here
			for (; t < tfar; t += stepsize) {
#ifdef SYCL
        float3 tmp{origin + direction * t};
				f_tt = volume.interp(tmp);   // interp is nonconst
#else
				f_tt = volume.interp(origin + direction * t);
#endif
				if (f_tt < 0)                  // got it, jump out of inner loop
					break;
				if (f_tt < 0.8f)               // coming closer, reduce stepsize
					stepsize = step;
				f_t = f_tt;
			}
			if (f_tt < 0) {           // got it, calculate accurate intersection
				t = t + stepsize * f_tt / (f_t - f_tt);
				return make_float4(origin + direction * t, t);
			}
		}
	}
	return make_float4(0);

}

#ifdef SYCL
struct raycastKernel {

// Although T is instantiated float, pos3D and normal are targeting float3 data
// Actually, T is instantiated as float3. This is due to the declaration of
// ocl_vertex, which was given a float3 type, rather than the uptyped OpenCL buf
template <typename T, typename U>
static void kernel(item<2> ix, T *pos3D, T *normal, U *v_data,
                   const uint3 v_size, const float3 v_dim, const Matrix4 view,
                   const float nearPlane, const float farPlane,
                   const float step, const float largestep)
{
  /*const*/ Volume<U *> volume;//{v_size,v_dim,v_data};
  volume.data = &v_data[0]; volume.size = v_size; volume.dim = v_dim;
  uint2 pos{ix[0],ix[1]};
  const int sizex = ix.get_range()[0];

  /*const*/ float4 hit =
    raycast_sycl(volume, pos, view, nearPlane, farPlane, step, largestep);
  const float3 test{hit.x(),hit.y(),hit.z()}; // as_float3(hit);

  if (hit.w() > 0.0f) {
    pos3D[pos.x() + sizex * pos.y()] = test;
    float3 surfNorm = grad(test,volume);
    if (cl::sycl::length(surfNorm) == 0) {
      normal[pos.x() + sizex * pos.y()] = float3{INVALID,INVALID,INVALID};
    } else {
      normal[pos.x() + sizex * pos.y()] = cl::sycl::normalize(surfNorm);
    }
  } else {
    pos3D [pos.x() + sizex * pos.y()] = float3{0,0,0};
    normal[pos.x() + sizex * pos.y()] = float3{INVALID,INVALID,INVALID};
  }
}

}; // struct
#else
void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
		/*const*/ Volume integration, const Matrix4 view, const float nearPlane,
		const float farPlane, const float step, const float largestep) {
	TICK();
	unsigned int y;
#pragma omp parallel for \
	    shared(normal, vertex), private(y)
	for (y = 0; y < inputSize.y; y++)
		for (unsigned int x = 0; x < inputSize.x; x++) {

			uint2 pos = make_uint2(x, y);

			const float4 hit = raycast(integration, pos, view, nearPlane,
					farPlane, step, largestep);
			if (hit.w > 0.0) {
				vertex[pos.x + pos.y * inputSize.x] = make_float3(hit);
				float3 surfNorm = integration.grad(make_float3(hit));
				if (length(surfNorm) == 0) {
					//normal[pos] = normalize(surfNorm); // APN added
					normal[pos.x + pos.y * inputSize.x].x = INVALID;
				} else {
					normal[pos.x + pos.y * inputSize.x] = normalize(surfNorm);
				}
			} else {
				//std::cerr<< "RAYCAST MISS "<<  pos.x << " " << pos.y <<"  " << hit.w() <<"\n";
				vertex[pos.x + pos.y * inputSize.x] = make_float3(0);
				normal[pos.x + pos.y * inputSize.x] = make_float3(INVALID,0,0);
			}
		}
	TOCK("raycastKernel", inputSize.x * inputSize.y);
}
#endif

bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold)
{
	bool res = false;
	TICK();
	// Update the pose regarding the tracking result
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	// pose = toMatrix4(delta) * pose; // * is nonconst; toMatrix4 is an rvalue
  auto tmp = toMatrix4(delta);
	pose = tmp * pose;

	// Return validity test result of the tracking
	if (norm(x) < icp_threshold)
		res = true;

	TOCK("updatePoseKernel", 1);
	return res;
}

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output,
		uint2 imageSize, float track_threshold) {

	// Check the tracking result, and go back to the previous camera position if necessary

	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

	if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2)
#ifdef SYCL
			|| (values(0, 28) / (imageSize.x() * imageSize.y()) < track_threshold)) {
#else
			|| (values(0, 28) / (imageSize.x   * imageSize.y  ) < track_threshold)) {
#endif
		pose = oldPose;
		return false;
	} else {
		return true;
	}

}

#ifdef SYCL
void renderNormalKernel(uchar3* out, const float3* normal, uint2 normalSize) {
	TICK();
	unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
	for (y = 0; y < normalSize.y(); y++)
		for (unsigned int x = 0; x < normalSize.x(); x++) {
			uint pos = x + y * normalSize.x();
			float3 n = normal[pos];
			if (n.x() == -2) {
				out[pos] = make_uchar3(0, 0, 0);
			} else {
				n = normalize(n);
				out[pos] = make_uchar3(n.x()*128+128, n.y()*128+128, n.z()*128+128);
			}
		}
	TOCK("renderNormalKernel", normalSize.x() * normalSize.y());
}
#else
void renderNormalKernel(uchar3* out, const float3* normal, uint2 normalSize) {
	TICK();
	unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
	for (y = 0; y < normalSize.y; y++)
		for (unsigned int x = 0; x < normalSize.x; x++) {
			uint pos = x + y * normalSize.x;
			float3 n = normal[pos];
			if (n.x == -2) {
				out[pos] = make_uchar3(0, 0, 0);
			} else {
				n = normalize(n);
				out[pos] = make_uchar3(n.x*128+128, n.y*128+128, n.z*128+128);
			}
		}
	TOCK("renderNormalKernel", normalSize.x * normalSize.y);
}
#endif

#ifdef SYCL
struct renderDepthKernel {

template <typename T, typename U>   // templates abstract over address spaces
static void kernel(item<2> ix, T *out, U const *depth,
                   const float nearPlane, const float farPlane)
{
	const int posx = ix[0];
	const int posy = ix[1];
  const int sizex = ix.get_range()[0];
	float d = depth[posx + sizex * posy];
	if (d < nearPlane)
    out[posx + sizex * posy] = uchar4{255,255,255,0};
	else {
		if (d > farPlane)
      out[posx + sizex * posy] = uchar4{0,0,0,0};
		else {
			float h = (d - nearPlane) / (farPlane - nearPlane);
			h *= 6.0f;
			const int sextant = (int)h;
			const float fract = h - sextant;
			const float mid1  = 0.25f + (0.5f*fract);
			const float mid2  = 0.75f - (0.5f*fract);
			switch (sextant)
			{
        case 0: out[posx + sizex * posy] = uchar4{191, 255*mid1, 64, 0}; break;
        case 1: out[posx + sizex * posy] = uchar4{255*mid2, 191, 64, 0}; break;
        case 2: out[posx + sizex * posy] = uchar4{64, 191, 255*mid1, 0}; break;
        case 3: out[posx + sizex * posy] = uchar4{64, 255*mid2, 191, 0}; break;
        case 4: out[posx + sizex * posy] = uchar4{255*mid1, 64, 191, 0}; break;
        case 5: out[posx + sizex * posy] = uchar4{191, 64, 255*mid2, 0}; break;
			}
		}
	}
}

}; // struct renderDepthKernel
#else
void renderDepthKernel(uchar4* out, float * depth, uint2 depthSize,
		const float nearPlane, const float farPlane) {
	TICK();

	float rangeScale = 1 / (farPlane - nearPlane);

	unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
	for (y = 0; y < depthSize.y; y++) {
		int rowOffeset = y * depthSize.x;
		for (unsigned int x = 0; x < depthSize.x; x++) {

			unsigned int pos = rowOffeset + x;

			if (depth[pos] < nearPlane)
				out[pos] = make_uchar4(255, 255, 255, 0); // The forth value is a padding in order to align memory
			else {
				if (depth[pos] > farPlane)
					out[pos] = make_uchar4(0, 0, 0, 0); // The forth value is a padding in order to align memory
				else {
					const float d = (depth[pos] - nearPlane) * rangeScale;
					out[pos] = gs2rgb(d);
				}
			}
		}
	}
	TOCK("renderDepthKernel", depthSize.x * depthSize.y);
}
#endif

#ifdef SYCL
struct renderTrackKernel {

template <typename T, typename U>
static void kernel(item<2> ix, T * out, const U * data) {

	const int posx  = ix[0];
	const int posy  = ix[1];
  const int sizex = ix.get_range()[0];

	switch (data[posx + sizex * posy].result) {
		case  1: out[posx + sizex * posy] = uchar4{128, 128, 128, 0}; break;
		case -1: out[posx + sizex * posy] = uchar4{000, 000, 000, 0}; break;
		case -2: out[posx + sizex * posy] = uchar4{255, 000, 000, 0}; break;
		case -3: out[posx + sizex * posy] = uchar4{000, 255, 000, 0}; break;
		case -4: out[posx + sizex * posy] = uchar4{000, 000, 255, 0}; break;
		case -5: out[posx + sizex * posy] = uchar4{255, 255, 000, 0}; break;
		default: out[posx + sizex * posy] = uchar4{255, 128, 128, 0}; break;
	}
}

}; // struct

#else
void renderTrackKernel(uchar4* out, const TrackData* data, uint2 outSize) {
	TICK();

	unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
	for (y = 0; y < outSize.y; y++)
		for (unsigned int x = 0; x < outSize.x; x++) {
			uint pos = x + y * outSize.x;
			switch (data[pos].result) {
			case 1:
				out[pos] = make_uchar4(128, 128, 128, 0);  // ok	 GREY
				break;
			case -1:
				out[pos] = make_uchar4(0, 0, 0, 0);      // no input BLACK
				break;
			case -2:
				out[pos] = make_uchar4(255, 0, 0, 0);        // not in image RED
				break;
			case -3:
				out[pos] = make_uchar4(0, 255, 0, 0);    // no correspondence GREEN
				break;
			case -4:
				out[pos] = make_uchar4(0, 0, 255, 0);        // to far away BLUE
				break;
			case -5:
				out[pos] = make_uchar4(255, 255, 0, 0);     // wrong normal YELLOW
				break;
			default:
				out[pos] = make_uchar4(255, 128, 128, 0);
				break;
			}
		}
	TOCK("renderTrackKernel", outSize.x * outSize.y);
}
#endif

#ifdef SYCL
struct renderVolumeKernel {

template <typename T, typename U>
static void kernel(item<2> ix, T *render, U *v_data, const uint3 v_size,
                   const float3 v_dim, const Matrix4 view,
                   const float nearPlane, const float farPlane,
                   const float step, const float largestep,
                   const float3 light, const float3 ambient)
{
	/*const*/ Volume<U *> v;//{v_size,v_dim,v_data};
  v.data = v_data; v.size = v_size; v.dim = v_dim;

	/*const*/ uint2 pos{ix[0],ix[1]};
  const int sizex = ix.get_range()[0];

  float4 hit = raycast(v, pos, view, nearPlane, farPlane,step, largestep);

	if (hit.w() > 0) {
    const float3 test{hit.x(),hit.y(),hit.z()}; // as_float3(hit);
		float3 surfNorm   = grad(test,v);

		if (length(surfNorm) > 0) {
      const float3 diff    = normalize(light - test);
      const float dir      = fmaxf(dot(normalize(surfNorm), diff), 0.f);
      /*const*/ float3 col = clamp(make_float3(dir)+ambient,0.f,1.f) * 255;
      render[pos.x() + sizex * pos.y()] = uchar4{col.x(),col.y(),col.z(),0};
		} else {
      render[pos.x() + sizex * pos.y()] = uchar4{0,0,0,0};
		}
	} else {
      render[pos.x() + sizex * pos.y()] = uchar4{0,0,0,0};
	}
}

}; // struct
#else
void renderVolumeKernel(uchar4* out, const uint2 depthSize, const Volume volume,
    const Matrix4 view, const float nearPlane,
    const float farPlane, const float step, const float largestep,
    const float3 light, const float3 ambient) {
	TICK();
	unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
	for (y = 0; y < depthSize.y; y++) {
		for (unsigned int x = 0; x < depthSize.x; x++) {
			const uint pos = x + y * depthSize.x;

			float4 hit = raycast(volume, make_uint2(x, y), view, nearPlane,
					farPlane, step, largestep);
			if (hit.w > 0) {
				const float3 test     = make_float3(hit);
				const float3 surfNorm = volume.grad(test);
				if (length(surfNorm) > 0) {
					const float3 diff = normalize(light - test);
					const float dir   = fmaxf(dot(normalize(surfNorm), diff), 0.f);
					const float3 col  = clamp(make_float3(dir)+ambient,0.f,1.f) * 255;
					out[pos] = make_uchar4(col.x, col.y, col.z, 0);       // arg 4 = pad
				} else {
					out[pos] = make_uchar4(0, 0, 0, 0);                   // ""
				}
			} else {
				out[pos] = make_uchar4(0, 0, 0, 0);                     // ""
			}
		}
	}
	TOCK("renderVolumeKernel", depthSize.x * depthSize.y);
}
#endif

template <typename T>
void dbg_show(T p, const char *fname, size_t sz, int id)
{
  typename std::remove_reference<decltype(p[0])>::type total{0};
//  decltype(p[0]) total{0};
  for (size_t i = 0; i < sz; i++)
    total += p[i];
  printf("(%d) sum of %s: %g\n", id, fname, total);
}
template <typename T>
void dbg_show2(T p, const char *fname, size_t sz, int id)
{
  short total{0};
  for (size_t i = 0; i < sz; i++)
#ifdef SYCL
    total += p[i].x() + p[i].y();
#else
    total += p[i].x   + p[i].y;
#endif
  printf("(%d) sum of %s: %hd\n", id, fname, total);
}
template <typename T>
void dbg_show3(T p, const char *fname, size_t sz, int id)
{
  float total{0};
  for (size_t i = 0; i < sz; i++)
#ifdef SYCL
    total += p[i].x() + p[i].y() + p[i].z();
#else
    total += p[i].x   + p[i].y   + p[i].z;
#endif
  printf("(%d) sum of %s: %g\n", id, fname, total);
}
template <typename T>
void dbg_show4(T p, const char *fname, size_t sz, int id)
{
  long total{0};
  for (size_t i = 0; i < sz; i++)
#ifdef SYCL
    total += p[i].x() + p[i].y() + p[i].z() + p[i].w();
#else
    total += p[i].x   + p[i].y   + p[i].z   + p[i].w;
#endif
  printf("(%d) sum of %s: %ld\n", id, fname, total);
}

template <typename T>
void dbg_show_TrackData(T p, const char *fname, size_t sz, int id)
{
  float total{0};
  for (size_t i = 0; i < sz; i++)
    total += p[i].J[0] + p[i].J[1] + p[i].J[2] +
             p[i].J[3] + p[i].J[4] + p[i].J[5];
  printf("(%d) sum of %s: %g\n", id, fname, total);
}

#ifdef SYCL
template <typename T>
void copy_back(T *p, buffer<T,1> &buf) {
  const auto  r = sycl_a::mode::read;
  const auto hb = sycl_a::target::host_buffer;
  auto ha = buf.template get_access<r, hb>();
  const range<1> extents = buf.get_range();

  const size_t e1 = extents[0];
  for (size_t i = 0; i < e1; ++i) {
    p[i] = ha[i];
  }
}
#endif

bool Kfusion::preprocessing(const uint16_t * inputDepth, /*const*/ uint2 inSize) {

	// bilateral_filter(ScaledDepth[0], inputDepth, inputSize , gaussian, e_delta, radius);
	uint2 outSize = computationSize;
#ifdef SYCL
	const uint inSize_x  = inSize.x(); const uint inSize_y = inSize.y();
	const uint outSize_x = computationSize.x();
  const uint outSize_y = computationSize.y();
#else
	const uint inSize_x  = inSize.x;   const uint inSize_y  = inSize.y;
	const uint outSize_x = computationSize.x;
  const uint outSize_y = computationSize.y;
#endif

	// Check for unsupported conditions
	if ((inSize_x < outSize_x) || (inSize_y < outSize_y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize_x % outSize_x != 0) || (inSize_y % outSize_y != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize_x / outSize_x != inSize_y / outSize_y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = inSize_x / outSize_x;

#ifdef SYCL
  auto sd0a = ocl_ScaledDepth[0]->get_access<
    sycl_a::mode::read,
    sycl_a::target::host_buffer
  >();
  dbg_show(sd0a, "ScaledDepth[0]", outSize_x * outSize_y, 0);
#else
  dbg_show(ScaledDepth[0], "ScaledDepth[0]", outSize_x * outSize_y, 0);
#endif
    
#ifdef SYCL
  {
    const range<1>  in_size{inSize.x()*inSize.y()};
    const range<1> out_size{outSize.x()*outSize.y()};
    // The const_casts overcome a SYCL buffer ctor bug causing a segfault
    buffer<ushort,1> ocl_depth_buffer(const_cast<ushort*>(inputDepth), in_size);
//    buffer< float,1> ocl_FloatDepth(out_size);
    buffer<decltype(ratio),1>   buf_ratio(&ratio,range<1>{sizeof(ratio)});
    buffer<decltype(outSize),1> buf_os(&outSize,range<1>{sizeof(outSize)});
    buffer<decltype(inSize),1>  buf_is(&inSize,range<1>{sizeof(inSize)});
    q.submit([&](handler &cgh) {

      auto in      = ocl_depth_buffer.get_access<sycl_a::mode::read      >(cgh);
      auto depth   =  ocl_FloatDepth->get_access<sycl_a::mode::read_write>(cgh);
      auto a_ratio   = buf_ratio.get_access<sycl_a::mode::read>(cgh); //
      auto a_outSize = buf_os.get_access<sycl_a::mode::read>(cgh); //
      auto a_inSize  = buf_is.get_access<sycl_a::mode::read>(cgh); //

      cgh.parallel_for<class T0>(range<2>{outSize.x(),outSize.y()},
        [in,depth,a_ratio,a_inSize,a_outSize](item<2> ix) {
        auto &ratio   = a_ratio  [0]; //
        auto &outSize = a_outSize[0]; //
        auto &inSize  = a_inSize [0]; //
        depth[ix[0] + outSize.x() * ix[1]] =
           in[ix[0] * ratio + inSize.x() * ix[1] * ratio] / 1000.0f;
//        depth[ix] = in[ ix.get()*ratio ] / 1000.0f;
      });
    });

    const size_t gaussianS = radius * 2 + 1;
//    buffer<float,1> ocl_ScaledDepth(ScaledDepth[0],out_size); // remove arg 1
    buffer<float,1> ocl_gaussian(gaussian, range<1>{gaussianS});
    decltype(radius)  stack_radius  = radius; 
    decltype(e_delta) stack_e_delta = e_delta;
    buffer<int,1>    buf_radius(const_cast<int*>(&stack_radius),range<1>{1});
    buffer<float,1> buf_e_delta(const_cast<float*>(&stack_e_delta),range<1>{1});

    q.submit([&](handler &cgh) {

      auto out       = ocl_ScaledDepth[0]->get_access<sycl_a::mode::write>(cgh);
      auto in        = ocl_FloatDepth->get_access<sycl_a::mode::read>(cgh);
      auto gaussian  =    ocl_gaussian.get_access<sycl_a::mode::read>(cgh);
      auto a_radius  =      buf_radius.get_access<sycl_a::mode::read>(cgh); //
      auto a_e_delta =     buf_e_delta.get_access<sycl_a::mode::read>(cgh); //
   
      cgh.parallel_for<class T1>(range<2>{outSize.x(),outSize.y()},
        [in,out,gaussian,a_radius,a_e_delta](item<2> ix) { 
          /*const*/ uint2 pos{ix[0],ix[1]};
          /*const*/ uint2 size{ix.get_range()[0], ix.get_range()[1]};
          auto &r   =  a_radius[0]; //
          auto &e_d = a_e_delta[0]; //

          const float center = in[pos.x() + size.x() * pos.y()];

          if ( center == 0 ) {
            out[pos.x() + size.x() * pos.y()] = 0;
            return;
          }

          float sum = 0.0f;
          float t   = 0.0f;
          for (int i = -r; i <= r; ++i) {
            for (int j = -r; j <= r; ++j) {
              // n.b. unsigned + signed is unsigned! Bug in OpenCL C version?
              const int px = pos.x()+i; const int sx = size.x()-1;
              const int py = pos.y()+i; const int sy = size.y()-1;
              const int curPosx = cl::sycl::clamp(px,0,sx);
              const int curPosy = cl::sycl::clamp(py,0,sy);
              const float curPix = in[curPosx + curPosy * size.x()];
              if (curPix > 0) {
                const float mod    = sq(curPix - center);
                const float factor = gaussian[i + r] * gaussian[j + r] *
                                     cl::sycl::exp(-mod / (2 * e_d * e_d));
                t   += factor * curPix;
                sum += factor;
              } else {
                // std::cerr << "ERROR BILATERAL " << pos.x()+i << " " <<
                // pos.y()+j<< " " <<curPix<<" \n";
              }
            }
          } 
          out[pos.x() + size.x() * pos.y()] = t / sum;
      }); 
    });

  }
  auto sd0b = ocl_ScaledDepth[0]->get_access<
    sycl_a::mode::read,
    sycl_a::target::host_buffer
  >();
  dbg_show(sd0b, "ScaledDepth[0]", outSize.x() * outSize.y(), 1);
#else

  mm2metersKernel(floatDepth, computationSize, inputDepth, inSize);
  bilateralFilterKernel(ScaledDepth[0], floatDepth, computationSize, gaussian,
    e_delta, radius);

  dbg_show(ScaledDepth[0], "ScaledDepth[0]", outSize_x * outSize_y, 1);
#endif


/*__kernel void mm2metersKernel(
		__global float * depth,
		const uint2 depthSize ,
		const __global ushort * in ,
		const uint2 inSize ,
		const int ratio ) {
	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	depth[pixel.x + depthSize.x * pixel.y()] = in[pixel.x * ratio + inSize.x * pixel.y() * ratio] / 1000.0f;
}
*/

/*__kernel void bilateralFilterKernel( __global float * out,
		const __global float * in,
		const __global float * gaussian,
		const float e_d,
		const int r ) {

	const uint2 pos = (uint2) (get_global_id(0),get_global_id(1));
	const uint2 size = (uint2) (get_global_size(0),get_global_size(1));

	const float center = in[pos.x + size.x * pos.y()];

	if ( center == 0 ) {
		out[pos.x + size.x * pos.y()] = 0;
		return;
	}

	float sum = 0.0f;
	float t = 0.0f;
	// FIXME : sum and t diverge too much from cpp version
	for(int i = -r; i <= r; ++i) {
		for(int j = -r; j <= r; ++j) {
			const uint2 curPos = (uint2)(clamp(pos.x + i, 0u, size.x-1), clamp(pos.y() + j, 0u, size.y()-1));
			const float curPix = in[curPos.x + curPos.y() * size.x];
			if(curPix > 0) {
				const float mod = sq(curPix - center);
				const float factor = gaussian[i + r] * gaussian[j + r] * exp(-mod / (2 * e_d * e_d));
				t += factor * curPix;
				sum += factor;
			} else {
				//std::cerr << "ERROR BILATERAL " <<pos.x+i<< " "<<pos.y()+j<< " " <<curPix<<" \n";
			}
		}
	}
	out[pos.x + size.x * pos.y()] = t / sum;

} */

//  mm2metersKernel(floatDepth, computationSize, inputDepth, inSize);
//  bilateralFilterKernel(ScaledDepth[0], floatDepth, computationSize, gaussian,
//    e_delta, radius);

	return true;
}

bool Kfusion::tracking(float4 k, float icp_threshold, uint tracking_rate,
		uint frame) {

	if (frame % tracking_rate != 0)
		return false;

	// half sample the input depth maps into the pyramid levels
	for (unsigned int i = 1; i < iterations.size(); ++i) {
#ifdef SYCL
		cl::sycl::uint2 outSize{computationSize.x() / (int) ::pow(2, i),
                            computationSize.y() / (int) ::pow(2, i)};

    auto r   = range<2>{outSize.x(),outSize.y()};
    auto out = dagr::wo(*ocl_ScaledDepth[i]);
		uint2 inSize{outSize.x()*2,outSize.y()*2}; // Seems redundant
    auto &in = *const_cast<const buffer<float,1> *>(ocl_ScaledDepth[i-1]);
    dagr::run<halfSampleRobustImageKernel,0>(q,r,out,in,inSize,e_delta*3,1);
#else
		halfSampleRobustImageKernel(ScaledDepth[i], ScaledDepth[i - 1],
				make_uint2(computationSize.x / (int) pow(2, i - 1),
						computationSize.y / (int) pow(2, i - 1)), e_delta * 3, 1);
#endif
	}
#ifdef SYCL
  auto sd0 = ocl_ScaledDepth[0]->get_access<
    sycl_a::mode::read,
    sycl_a::target::host_buffer
  >();
  auto sd1 = ocl_ScaledDepth[1]->get_access<
    sycl_a::mode::read,
    sycl_a::target::host_buffer
  >();
  dbg_show(sd0, "ScaledDepth[0]", (computationSize.x() * computationSize.y()) / (int)pow(2,0), 2);
  dbg_show(sd1, "ScaledDepth[1]", (computationSize.x() * computationSize.y()) / (int)pow(2,1), 2);
#else
  dbg_show(ScaledDepth[0], "ScaledDepth[0]", (computationSize.x   * computationSize.y) / (int)pow(2,0), 2);
  dbg_show(ScaledDepth[1], "ScaledDepth[1]", (computationSize.x * computationSize.y) / (int)pow(2,1), 2);
#endif

	// prepare the 3D information from the input depth maps
	uint2 localimagesize = computationSize;
	for (unsigned int i = 0; i < iterations.size(); ++i) {
#ifdef SYCL
		float4 tmp{k / float(1 << i)};
		Matrix4 invK = getInverseCameraMatrix(tmp);   // Needs a non-const (tmp)
    const uint2    img_sz_ui2{localimagesize.x(),localimagesize.y()};
    const range<2>  imageSize{localimagesize.x(),localimagesize.y()};
    auto &depth  = *const_cast<const buffer<float,1> *>(ocl_ScaledDepth[i]);
    dagr::run<depth2vertexKernel,0>(q,imageSize,
      *ocl_inputVertex[i],img_sz_ui2,depth,img_sz_ui2,invK);

    const range<2>  imageSize2{localimagesize.x(),localimagesize.y()};
    auto &vertex = *const_cast<const buffer<float3,1> *>(ocl_inputVertex[i]);
    dagr::run<vertex2normalKernel,0>(q,imageSize2,
      *ocl_inputNormal[i],img_sz_ui2,vertex,img_sz_ui2);

		localimagesize = make_uint2(localimagesize.x() / 2, localimagesize.y() / 2);
#else
		Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));
		depth2vertexKernel(inputVertex[i], ScaledDepth[i], localimagesize, invK);
		vertex2normalKernel(inputNormal[i], inputVertex[i], localimagesize);
		localimagesize = make_uint2(localimagesize.x / 2, localimagesize.y / 2);
#endif
	}
#ifdef SYCL
  auto iv = ocl_inputVertex[0]->get_access<
    sycl_a::mode::read,
    sycl_a::target::host_buffer
  >();
  auto in = ocl_inputNormal[0]->get_access<
    sycl_a::mode::read,
    sycl_a::target::host_buffer
  >();
  dbg_show3(iv, "inputVertex[0]", (computationSize.x() * computationSize.y()) / (int)pow(2,0), 3);
  dbg_show3(in, "inputNormal[0]", (computationSize.x() * computationSize.y()) / (int)pow(2,0), 4);
#else
  dbg_show3(inputVertex[0], "inputVertex[0]", (computationSize.x * computationSize.y) / (int)pow(2,0), 3);
  dbg_show3(inputNormal[0], "inputNormal[0]", (computationSize.x * computationSize.y) / (int)pow(2,0), 4);
#endif

	oldPose = pose;
	const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

  // iterations: a vector<int> set to {10,5,4} in Kfusion ctor (kernels.h)
  for (int level = iterations.size() - 1; level >= 0; --level) {
#ifdef SYCL
    const int csize_x = computationSize.x();
    const int csize_y = computationSize.y();
#else
    const int csize_x = computationSize.x;
    const int csize_y = computationSize.y;
#endif
    const int pow2l = pow(2, level);
		uint2 localimagesize = make_uint2(csize_x / pow2l, csize_y / pow2l);

    for (int i = 0; i < iterations[level]; ++i) {  // i<4,i<5,i<10
#ifdef SYCL
      using cbf3_t = const buffer<float3,1>;
      auto &inVertex  = *const_cast<cbf3_t *>(ocl_inputVertex[level]);
      auto &inNormal  = *const_cast<cbf3_t *>(ocl_inputNormal[level]);
      auto &refVertex = *const_cast<cbf3_t *>(ocl_vertex);
      auto &refNormal = *const_cast<cbf3_t *>(ocl_normal);
      range<2> imageSize{localimagesize.x(),localimagesize.y()};
      dagr::run<trackKernel,0>(q,imageSize,
        *ocl_trackingResult,computationSize,
        inVertex,localimagesize,inNormal,localimagesize,
        refVertex,computationSize,refNormal,computationSize,
        pose,projectReference,dist_threshold,normal_threshold);

      const auto rw = sycl_a::mode::read_write; // previously above
      const    range<1> nitems{size_of_group * number_of_groups};
      const nd_range<1> ndr{nd_range<1>(nitems, range<1>{size_of_group})};
      uint2 JSize{computationSize.x(), computationSize.y()};
      uint2  size{ localimagesize.x(),  localimagesize.y()};
      buffer<uint2,1> buf_JSize(&JSize,range<1>{1});
      buffer<uint2,1> buf_size(  &size,range<1>{1});

      q.submit([&](handler &cgh) {
        auto       J = ocl_trackingResult->get_access<sycl_a::mode::read>(cgh);
        auto a_JSize = buf_JSize.get_access<sycl_a::mode::read>(cgh);
        auto  a_size =  buf_size.get_access<sycl_a::mode::read>(cgh);
        const range<1> local_mem_size{size_of_group * 32};
        auto out=ocl_reduce_output_buffer->get_access<sycl_a::mode::write>(cgh);
        accessor<float, 1, rw, sycl_a::target::local> S(local_mem_size, cgh);

        cgh.parallel_for<class T6>(ndr,[out,J,a_JSize,a_size,S](nd_item<1> ix) {
          auto &JSize = a_JSize[0];  //
          auto  &size =  a_size[0];  //
          uint blockIdx  = ix.get_group(0);
          uint blockDim  = ix.get_local_range(0);
          uint threadIdx = ix.get_local(0);
          //uint gridDim   = ix.get_num_groups(0); // bug: always 0
          uint gridDim   = ix.get_global_range(0) / ix.get_local_range(0);

          const uint sline = threadIdx;

          float         sums[32];
          float *jtj  = sums + 7;
          float *info = sums + 28;

          for (uint i = 0; i < 32; ++i)
            sums[i] = 0.0f;

          // Is gridDim zero!?
          for (uint y = blockIdx; y < size.y(); y += gridDim) {
            for (uint x = sline; x < size.x(); x += blockDim) {
              const TrackData row = J[x + y * JSize.x()];
              if (row.result < 1) {
                info[1] += row.result == -4 ? 1 : 0;
                info[2] += row.result == -5 ? 1 : 0;
                info[3] += row.result > -4 ? 1 : 0;
                continue;
              }

              // Error part
              sums[0] += row.error * row.error;

              // JTe part
              for (int i = 0; i < 6; ++i)
                sums[i+1] += row.error * row.J[i];

              jtj[0] += row.J[0] * row.J[0];
              jtj[1] += row.J[0] * row.J[1];
              jtj[2] += row.J[0] * row.J[2];
              jtj[3] += row.J[0] * row.J[3];
              jtj[4] += row.J[0] * row.J[4];
              jtj[5] += row.J[0] * row.J[5];

              jtj[6] += row.J[1] * row.J[1];
              jtj[7] += row.J[1] * row.J[2];
              jtj[8] += row.J[1] * row.J[3];
              jtj[9] += row.J[1] * row.J[4];
              jtj[10] += row.J[1] * row.J[5];

              jtj[11] += row.J[2] * row.J[2];
              jtj[12] += row.J[2] * row.J[3];
              jtj[13] += row.J[2] * row.J[4];
              jtj[14] += row.J[2] * row.J[5];

              jtj[15] += row.J[3] * row.J[3];
              jtj[16] += row.J[3] * row.J[4];
              jtj[17] += row.J[3] * row.J[5];

              jtj[18] += row.J[4] * row.J[4];
              jtj[19] += row.J[4] * row.J[5];

              jtj[20] += row.J[5] * row.J[5];
              // extra info here
              info[0] += 1;
            }
          }

          // copy over to shared memory
          for (int i = 0; i < 32; ++i)
            S[sline * 32 + i] = sums[i];

          ix.barrier(sycl_a::fence_space::local);

          // sum up columns and copy to global memory in the final 32 threads
          if (sline < 32) {
            for (unsigned i = 1; i < blockDim; ++i)
              S[sline] += S[i * 32 + sline];
            out[sline+blockIdx*32] = S[sline];
          }
        });
      });

//      copy_back(reduceOutputBuffer, *ocl_reduce_output_buffer);
//      memcpy(reductionoutput,
//             reduceOutputBuffer,
//             number_of_groups * 32 * sizeof(float));

      if (updatePoseKernel(pose, reduceOutputBuffer, icp_threshold))
        break;
#else
			trackKernel(trackingResult, inputVertex[level], inputNormal[level],
					localimagesize, vertex, normal, computationSize, pose,
					projectReference, dist_threshold, normal_threshold);

			reduceKernel(reductionoutput, trackingResult, computationSize,
					localimagesize);

			if (updatePoseKernel(pose, reductionoutput, icp_threshold))
				break;
#endif

    }           // for (int i = 0; i < iterations[level]; ++i)
  }             // for (int level = iterations.size() - 1; level >= 0; --level) 
#ifdef SYCL
  auto tres = ocl_trackingResult->get_access<
    sycl_a::mode::read,
    sycl_a::target::host_buffer
  >();
  dbg_show_TrackData(tres, "trackingResult",
                     computationSize.x() * computationSize.y(), 5);

  auto red = ocl_reduce_output_buffer->get_access<
    sycl_a::mode::read,
    sycl_a::target::host_buffer
  >();
  dbg_show(red,"reduceOutputBuffer",32*number_of_groups/*8*/,6);
	return checkPoseKernel(pose, oldPose, reduceOutputBuffer, computationSize,
			track_threshold);
#else
  dbg_show_TrackData(trackingResult, "trackingResult",
                     computationSize.x * computationSize.y, 5);
  dbg_show(reductionoutput,"reductionoutput",32*8,6);
	return checkPoseKernel(pose, oldPose, reductionoutput, computationSize,
			track_threshold);
#endif
}

#ifdef SYCL
template <typename T>
inline float vs(/*const*/ uint3 pos, /*const*/ Volume<T> v) {
	return v.data[pos.x() +
                pos.y() * v.size.x() +
                pos.z() * v.size.x() * v.size.y()].x();
}

template <typename T>
inline float interp(/*const*/ float3 pos, /*const*/ Volume<T> v) {
	const float3 scaled_pos = {(pos.x() * v.size.x() / v.dim.x()) - 0.5f,
                             (pos.y() * v.size.y() / v.dim.y()) - 0.5f,
                             (pos.z() * v.size.z() / v.dim.z()) - 0.5f};
//	float3 basef{0,0,0};
  float3 tmp = cl::sycl::floor(scaled_pos);
	const int3 base{tmp.x(),tmp.y(),tmp.z()};
//	const float3 factor{cl::sycl::fract(scaled_pos, (float3 *) &basef)};
  /*const*/ float3 factor =
    cl::sycl::fmin(scaled_pos - cl::sycl::floor(scaled_pos), 0x1.fffffep-1f);
  //float3 basef = cl::sycl::floor(scaled_pos);

  /*const*/ int3 lower = max(base, int3{0,0,0});
  /*const*/ int3 upper = min(base + int3{1,1,1},
                             int3{v.size.x()-1,v.size.y()-1,v.size.z()-1});
	return (((vs(uint3{lower.x(), lower.y(), lower.z()}, v) * (1 - factor.x())
			+ vs(uint3{upper.x(), lower.y(), lower.z()}, v) * factor.x())
			* (1 - factor.y())
			+ (vs(uint3{lower.x(), upper.y(), lower.z()}, v) * (1 - factor.x())
					+ vs(uint3{upper.x(), upper.y(), lower.z()}, v) * factor.x())
					* factor.y()) * (1 - factor.z())
			+ ((vs(uint3{lower.x(), lower.y(), upper.z()}, v) * (1 - factor.x())
					+ vs(uint3{upper.x(), lower.y(), upper.z()}, v) * factor.x())
					* (1 - factor.y())
					+ (vs(uint3{lower.x(), upper.y(), upper.z()}, v)
							* (1 - factor.x())
							+ vs(uint3{upper.x(), upper.y(), upper.z()}, v)
									* factor.x()) * factor.y()) * factor.z())
			* 0.00003051944088f;
  return 0;
}

template <typename T>
inline float3 grad(float3 pos, /*const*/ Volume<T> v) {
	/*const*/ float3 scaled_pos = {(pos.x() * v.size.x() / v.dim.x()) - 0.5f,
                                 (pos.y() * v.size.y() / v.dim.y()) - 0.5f,
                                 (pos.z() * v.size.z() / v.dim.z()) - 0.5f};
	const int3 base{cl::sycl::floor(scaled_pos.x()),
                  cl::sycl::floor(scaled_pos.y()),
			            cl::sycl::floor(scaled_pos.z())};
	//const float3 basef{0,0,0};
	//const float3 factor = (float3) fract(scaled_pos, (float3 *) &basef);
  /*const*/ float3 factor =
    cl::sycl::fmin(scaled_pos - cl::sycl::floor(scaled_pos), 0x1.fffffep-1f);
  //float3 basef = cl::sycl::floor(scaled_pos);

  const int3 vsm1{static_cast<int>(v.size.x()) - 1,
                  static_cast<int>(v.size.y()) - 1,
                  static_cast<int>(v.size.z()) - 1};
	/*const*/ int3 lower_lower = max(base - int3{1,1,1}, int3{0,0,0});
	/*const*/ int3 lower_upper = max(base,               int3{0,0,0});
	/*const*/ int3 upper_lower = min(base + int3{1,1,1}, vsm1);
	/*const*/ int3 upper_upper = min(base + int3{2,2,2}, vsm1);
	/*const*/ int3 lower       = lower_upper;
	/*const*/ int3 upper       = upper_lower;

	float3 gradient;

	gradient.x() = (((vs(uint3{upper_lower.x(), lower.y(), lower.z()}, v)
			- vs(uint3{lower_lower.x(), lower.y(), lower.z()}, v)) * (1 - factor.x())
			+ (vs(uint3{upper_upper.x(), lower.y(), lower.z()}, v)
					- vs(uint3{lower_upper.x(), lower.y(), lower.z()}, v))
					* factor.x()) * (1 - factor.y())
			+ ((vs(uint3{upper_lower.x(), upper.y(), lower.z()}, v)
					- vs(uint3{lower_lower.x(), upper.y(), lower.z()}, v))
					* (1 - factor.x())
					+ (vs(uint3{upper_upper.x(), upper.y(), lower.z()}, v)
							- vs(uint3{lower_upper.x(), upper.y(), lower.z()}, v))
							* factor.x()) * factor.y()) * (1 - factor.z())
			+ (((vs(uint3{upper_lower.x(), lower.y(), upper.z()}, v)
					- vs(uint3{lower_lower.x(), lower.y(), upper.z()}, v))
					* (1 - factor.x())
					+ (vs(uint3{upper_upper.x(), lower.y(), upper.z()}, v)
							- vs(uint3{lower_upper.x(), lower.y(), upper.z()}, v))
							* factor.x()) * (1 - factor.y())
					+ ((vs(uint3{upper_lower.x(), upper.y(), upper.z()}, v)
							- vs(uint3{lower_lower.x(), upper.y(), upper.z()}, v))
							* (1 - factor.x())
							+ (vs(uint3{upper_upper.x(), upper.y(), upper.z()}, v)
									- vs(
											uint3{lower_upper.x(), upper.y(),
													upper.z()}, v)) * factor.x())
							* factor.y()) * factor.z();

	gradient.y() = (((vs(uint3{lower.x(), upper_lower.y(), lower.z()}, v)
			- vs(uint3{lower.x(), lower_lower.y(), lower.z()}, v)) * (1 - factor.x())
			+ (vs(uint3{upper.x(), upper_lower.y(), lower.z()}, v)
					- vs(uint3{upper.x(), lower_lower.y(), lower.z()}, v))
					* factor.x()) * (1 - factor.y())
			+ ((vs(uint3{lower.x(), upper_upper.y(), lower.z()}, v)
					- vs(uint3{lower.x(), lower_upper.y(), lower.z()}, v))
					* (1 - factor.x())
					+ (vs(uint3{upper.x(), upper_upper.y(), lower.z()}, v)
							- vs(uint3{upper.x(), lower_upper.y(), lower.z()}, v))
							* factor.x()) * factor.y()) * (1 - factor.z())
			+ (((vs(uint3{lower.x(), upper_lower.y(), upper.z()}, v)
					- vs(uint3{lower.x(), lower_lower.y(), upper.z()}, v))
					* (1 - factor.x())
					+ (vs(uint3{upper.x(), upper_lower.y(), upper.z()}, v)
							- vs(uint3{upper.x(), lower_lower.y(), upper.z()}, v))
							* factor.x()) * (1 - factor.y())
					+ ((vs(uint3{lower.x(), upper_upper.y(), upper.z()}, v)
							- vs(uint3{lower.x(), lower_upper.y(), upper.z()}, v))
							* (1 - factor.x())
							+ (vs(uint3{upper.x(), upper_upper.y(), upper.z()}, v)
									- vs(
											uint3{upper.x(), lower_upper.y(),
													upper.z()}, v)) * factor.x())
							* factor.y()) * factor.z();

	gradient.z() = (((vs(uint3{lower.x(), lower.y(), upper_lower.z()}, v)
			- vs(uint3{lower.x(), lower.y(), lower_lower.z()}, v)) * (1 - factor.x())
			+ (vs(uint3{upper.x(), lower.y(), upper_lower.z()}, v)
					- vs(uint3{upper.x(), lower.y(), lower_lower.z()}, v))
					* factor.x()) * (1 - factor.y())
			+ ((vs(uint3{lower.x(), upper.y(), upper_lower.z()}, v)
					- vs(uint3{lower.x(), upper.y(), lower_lower.z()}, v))
					* (1 - factor.x())
					+ (vs(uint3{upper.x(), upper.y(), upper_lower.z()}, v)
							- vs(uint3{upper.x(), upper.y(), lower_lower.z()}, v))
							* factor.x()) * factor.y()) * (1 - factor.z())
			+ (((vs(uint3{lower.x(), lower.y(), upper_upper.z()}, v)
					- vs(uint3{lower.x(), lower.y(), lower_upper.z()}, v))
					* (1 - factor.x())
					+ (vs(uint3{upper.x(), lower.y(), upper_upper.z()}, v)
							- vs(uint3{upper.x(), lower.y(), lower_upper.z()}, v))
							* factor.x()) * (1 - factor.y())
					+ ((vs(uint3{lower.x(), upper.y(), upper_upper.z()}, v)
							- vs(uint3{lower.x(), upper.y(), lower_upper.z()}, v))
							* (1 - factor.x())
							+ (vs(uint3{upper.x(), upper.y(), upper_upper.z()}, v)
									- vs(
											uint3{upper.x(), upper.y(),
													lower_upper.z()}, v))
									* factor.x()) * factor.y()) * factor.z();

	return gradient * float3{v.dim.x() / v.size.x(),
                           v.dim.y() / v.size.y(),
                           v.dim.z() / v.size.z()} * (0.5f * 0.00003051944088f);
}

template <typename T>
cl::sycl::float4 raycast_sycl(/*const*/ Volume<T>  v,
                         /*const*/ cl::sycl::uint2 pos,
                         const Matrix4 view, const float nearPlane,
                         const float farPlane, const float step,
                         const float largestep)
{
  const float3 origin = get_translation(view);
  /*const*/ float3 direction = myrotate(view, float3{pos.x(), pos.y(), 1.f});

	// intersect ray with a box
	//
	// www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
	// compute intersection of ray with all six bbox planes
  const float3 invR{1.0f/direction.x(), 1.0f/direction.y(), 1.0f/direction.z()};
  const float3 tbot = -1 * invR * origin;
	const float3 ttop = invR * (v.dim - origin);

  // re-order intersections to find smallest and largest on each axis
  /*const*/ float3 tmin = fmin(ttop, tbot);
  /*const*/ float3 tmax = fmax(ttop, tbot);

	// find the largest tmin and the smallest tmax
	const float largest_tmin  = fmax(fmax(tmin.x(), tmin.y()),
                                   fmax(tmin.x(), tmin.z()));
	const float smallest_tmax = fmin(fmin(tmax.x(), tmax.y()),
                                   fmin(tmax.x(), tmax.z()));

	// check against near and far plane
	const float tnear = fmax(largest_tmin, nearPlane);
	const float tfar  = fmin(smallest_tmax, farPlane);

  if (tnear < tfar) {
    // first walk with largesteps until we found a hit
    float t        = tnear;
    float stepsize = largestep;
    float f_t      = interp(origin + direction * t, v);
    float f_tt     = 0;
    if (f_t > 0) {  // oops, if we're already in it, don't render anything here
      for (; t < tfar; t += stepsize) {
        f_tt = interp(origin + direction * t, v);
        if (f_tt < 0)               // got it, jump out of inner loop
          break;
        if (f_tt < 0.8f)            // coming closer, reduce stepsize
          stepsize = step;
        f_t = f_tt;
      }
      if (f_tt < 0) {               // got it, calculate accurate intersection
        t = t + stepsize * f_tt / (f_t - f_tt);
        float3 tmp{origin + direction * t};
        return float4{tmp.x(),tmp.y(),tmp.z(),t};
      }
    }
  }

  return float4{0,0,0,0};
}
#endif

bool Kfusion::raycasting(float4 k, float mu, uint frame) {

	bool doRaycast = false;
	float largestep = mu * 0.75f;

	if (frame > 2) {
#ifdef SYCL
		raycastPose = pose;
		const Matrix4 view = raycastPose * getInverseCameraMatrix(k);
    range<2> RaycastglobalWorksize{computationSize.x(), computationSize.y()};

    dagr::run<raycastKernel,0>(q,RaycastglobalWorksize,
      *ocl_vertex,*ocl_normal,*ocl_volume_data,
      volumeResolution,volumeDimensions,view,nearPlane,farPlane,step,largestep);

    const auto csize = computationSize.x() * computationSize.y();
    auto a_vert = ocl_vertex->get_access<sycl_a::mode::read,
                                         sycl_a::target::host_buffer>();
    auto a_norm = ocl_normal->get_access<sycl_a::mode::read,
                                         sycl_a::target::host_buffer>();
    dbg_show3(a_vert, "vertex:   ", csize, 8);
    dbg_show3(a_norm, "normal:   ", csize, 9);
#else
		raycastPose = pose;
		raycastKernel(vertex, normal, computationSize, volume,
				raycastPose * getInverseCameraMatrix(k), nearPlane, farPlane,
				step, 0.75f * mu);
    const auto csize = computationSize.x * computationSize.y;
    dbg_show3(vertex, "vertex:   ", csize, 8);
    dbg_show3(normal, "normal:   ", csize, 9);
#endif
	}

	return doRaycast;
}

bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame)
{
#ifdef SYCL
	bool doIntegrate = checkPoseKernel(pose, oldPose, reduceOutputBuffer,
			computationSize, track_threshold);
#else
	bool doIntegrate = checkPoseKernel(pose, oldPose, reductionoutput,
			computationSize, track_threshold);
#endif

	if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
#ifdef SYCL
    range<2> globalWorksize{volumeResolution.x(), volumeResolution.y()};
    auto depth = *const_cast<buffer<float,1>*>(ocl_FloatDepth);
		const Matrix4 invTrack = inverse(pose);
		const Matrix4 K = getCameraMatrix(k);
		const float3 delta = myrotate(invTrack,
				float3{0, 0, volumeDimensions.z() / volumeResolution.z()});

    // The SYCL lambda for integrateKernel demonstrates verbosity
    dagr::run<integrateKernel,0>(q, globalWorksize,
      *ocl_volume_data, volumeResolution, volumeDimensions, depth,
      computationSize, invTrack, K, mu, maxweight, delta, myrotate(K, delta));

    const auto vsize = volume.size.x() * volume.size.y() * volume.size.z();
    auto vd = ocl_volume_data->get_access<sycl_a::mode::read,
                                          sycl_a::target::host_buffer>();
    dbg_show2(vd, "volume.data", vsize, 7);
#else
    integrateKernel(volume, floatDepth, computationSize, inverse(pose),
                    getCameraMatrix(k), mu, maxweight);
    const auto vsize = volume.size.x * volume.size.y * volume.size.z;
    dbg_show2(volume.data, "volume.data", vsize, 7); // short2 data
#endif
		doIntegrate = true;
	} else {
		doIntegrate = false;
	}

  printf("                    Kfusion::integration returns: %d\n", doIntegrate);
	return doIntegrate;
}

void Kfusion::dumpVolume(std::string filename) {

	std::ofstream fDumpFile;

	if (filename == "") {
		return;
	}

	std::cout << "Dumping the volumetric representation on file: " << filename
			<< std::endl;
	fDumpFile.open(filename.c_str(), std::ios::out | std::ios::binary);
	if (fDumpFile == NULL) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}

#ifdef SYCL
	const unsigned vol_size = volume.size.x() * volume.size.y() * volume.size.z();
#else
	const unsigned vol_size = volume.size.x   * volume.size.y   * volume.size.z;
#endif

	// Dump on file without the y component of the short2 variable
	for (unsigned int i = 0; i < vol_size; i++) {
		fDumpFile.write((char *) (volume.data + i), sizeof(short));
	}

	fDumpFile.close();

}

#ifdef SYCL
inline void update_ocl_output_render_buffer(/*const*/ uint2 &outputSize)
{
  if (outputImageSizeBkp.x() < outputSize.x() ||
      outputImageSizeBkp.y() < outputSize.y() ||
      ocl_output_render_buffer == NULL) 
  {
	  outputImageSizeBkp = make_uint2(outputSize.x(), outputSize.y());
	  if (ocl_output_render_buffer != NULL) {
	    delete ocl_output_render_buffer;
	    ocl_output_render_buffer = NULL;
	  }
	  ocl_output_render_buffer =
      new buffer<uchar4,1>(range<1>{outputSize.x() * outputSize.y()});
  }
}
#else
inline void update_ocl_output_render_buffer(const uint2 &) { }
#endif

void Kfusion::renderVolume(uchar4 * out, uint2 outputSize, int frame,
                           int raycast_rendering_rate, float4 k,
                           float largestep)
{
	if (frame % raycast_rendering_rate != 0) return;
  update_ocl_output_render_buffer(outputSize);

#ifdef SYCL
  range<2> globalWorksize{computationSize.x(), computationSize.y()};
  Matrix4 view = *(this->viewPose) * getInverseCameraMatrix(k);
  dagr::run<renderVolumeKernel,0>(q,globalWorksize,
    dagr::wo(*ocl_output_render_buffer),
    *ocl_volume_data,volumeResolution,volumeDimensions,view,nearPlane,farPlane,
    step,largestep,light,ambient);
#else
		renderVolumeKernel(out, outputSize, volume,
				*(this->viewPose) * getInverseCameraMatrix(k), nearPlane,
				farPlane * 2.0f, step, largestep, light, ambient); // why * by 2.0f ?
#endif
}

void Kfusion::renderTrack(uchar4 * out, uint2 outputSize) {
#ifdef SYCL
  update_ocl_output_render_buffer(outputSize);

  range<2> globalWorksize{computationSize.x(), computationSize.y()};
  dagr::run<renderTrackKernel,0>(q,globalWorksize,
    dagr::wo(*ocl_output_render_buffer),
    *const_cast<const buffer<TrackData,1> *>(ocl_trackingResult));

  const auto csize = computationSize.x() * computationSize.y();
  auto a_out = ocl_output_render_buffer->get_access<sycl_a::mode::read,
                                                 sycl_a::target::host_buffer>();
  dbg_show4(a_out, "trackRender", csize, 11);
#else
	renderTrackKernel(out, trackingResult, outputSize);

  const auto csize = computationSize.x * computationSize.y;
  dbg_show4(out, "trackRender", csize, 11);
#endif
}

void Kfusion::renderDepth(uchar4 * out, uint2 outputSize) {
#ifdef SYCL
  update_ocl_output_render_buffer(outputSize);

  range<2> globalWorksize{computationSize.x(), computationSize.y()};
  dagr::run<renderDepthKernel,0>(q,globalWorksize,
    dagr::wo(*ocl_output_render_buffer),
    *(const_cast<const decltype(ocl_FloatDepth)>(ocl_FloatDepth)),
    nearPlane, farPlane);

  const auto osize = outputSize.x() * outputSize.y();
  auto a_out = ocl_output_render_buffer->get_access<sycl_a::mode::read,
                                                 sycl_a::target::host_buffer>();
  dbg_show4(out, "depthRender: ", osize, 10);
#else
	renderDepthKernel(out, floatDepth, outputSize, nearPlane, farPlane);

  const auto osize = outputSize.x * outputSize.y;
  dbg_show4(out, "depthRender: ", osize, 10);
#endif
}

void synchroniseDevices() {
	// Nothing to do in the C++ implementation
}
