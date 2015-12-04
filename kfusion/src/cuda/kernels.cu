/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifdef __APPLE__
	#include <mach/clock.h>
	#include <mach/mach.h>

	#define TICK(str)    {static const std::string str_tick = str; \
		if (print_kernel_timing) {\
		    host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);\
		    clock_get_time(cclock, &tick_clockData);\
		    mach_port_deallocate(mach_task_self(), cclock);\
		}}

	#define TOCK()  {if (print_kernel_timing) {cudaDeviceSynchronize(); \
		host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);\
		clock_get_time(cclock, &tock_clockData);\
		mach_port_deallocate(mach_task_self(), cclock);\
		if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
		std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << std::endl;}}
#else
	
	#define TICK(str)    {static const std::string str_tick = str; \
		if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tick_clockData);}
	
	#define TOCK()        if (print_kernel_timing) {cudaDeviceSynchronize(); \
		clock_gettime(CLOCK_MONOTONIC, &tock_clockData); \
		std::cerr<< str_tick << " ";\
		if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec)) std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
		    std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << std::endl;}}

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

#include "kfusion.h"

#undef isnan
#undef isfinite

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/GR_SVD.h>

#include <constant_parameters.h>

#define INVALID -2   // this is used to mark invalid entries in normal or vertex maps

using namespace std;

__global__ void renderDepthKernel(Image<uchar4> out, const Image<float> depth,
		const float nearPlane, const float farPlane) {
	//const float d = (clamp(depth.el(), nearPlane, farPlane) - nearPlane) / (farPlane - nearPlane);
	//out.el() = make_uchar3(d * 255, d * 255, d * 255);
	if (depth.el() < nearPlane)
		out.el() = make_uchar4(255, 255, 255, 0); // The forth value is padding for memory alignement and so it is for following uchar4
	else {
		if (depth.el() > farPlane)
			out.el() = make_uchar4(0, 0, 0, 0); 
		else {
			float h = (depth.el() - nearPlane) / (farPlane - nearPlane);
			h *= 6.0;
			const int sextant = (int) h;
			const float fract = h - sextant;
			const float mid1 = 0.25 + (0.5 * fract);
			const float mid2 = 0.75 - (0.5 * fract);
			switch (sextant) {
			    case 0: out.el() = make_uchar4(191, 255 * mid1, 64, 0); break;
			    case 1: out.el() = make_uchar4(255 * mid2, 191, 64, 0); break;
			    case 2: out.el() = make_uchar4(64, 191, 255 * mid1, 0); break;
			    case 3: out.el() = make_uchar4(64, 255 * mid2, 191, 0); break;
			    case 4: out.el() = make_uchar4(255 * mid1, 64, 191, 0); break;
			    case 5: out.el() = make_uchar4(191, 64, 255 * mid2, 0); break;
			}
			// out.el() = gs2rgb(d);
		}
	}
}

__global__ void renderTrackKernel(Image<uchar4> out,
		const Image<TrackData> data) {
	const uint2 pos = thr2pos2();
	// The forth value is padding for memory alignement and so it is for following uchar4
	switch (data[pos].result) {
	    case  1: out[pos] = make_uchar4(128, 128, 128, 0);	break; // ok
	    case -1: out[pos] = make_uchar4(0, 0, 0, 0);	break; // no input 
	    case -2: out[pos] = make_uchar4(255, 0, 0, 0);	break; // not in image 
	    case -3: out[pos] = make_uchar4(0, 255, 0, 0);	break; // no correspondence
	    case -4: out[pos] = make_uchar4(0, 0, 255, 0);	break; // too far away
	    case -5: out[pos] = make_uchar4(255, 255, 0, 0);	break; // wrong normal
	}
}

__global__ void renderVolumeKernel(Image<uchar4> render, const Volume volume,
		const Matrix4 view, const float nearPlane, const float farPlane,
		const float step, const float largestep, const float3 light,
		const float3 ambient) {
	const uint2 pos = thr2pos2();

	float4 hit = raycast(volume, pos, view, nearPlane, farPlane, step,
			largestep);
	if (hit.w > 0) {
		const float3 test = make_float3(hit);
		const float3 surfNorm = volume.grad(test);
		if (length(surfNorm) > 0) {
			const float3 diff = normalize(light - test);
			const float dir = fmaxf(dot(normalize(surfNorm), diff), 0.f);
			const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f)
					* 255;
			render.el() = make_uchar4(col.x, col.y, col.z, 0); // The forth value is padding for memory alignement and so it is for following uchar4
		} else {
			render.el() = make_uchar4(0, 0, 0, 0);
		}
	} else {
		render.el() = make_uchar4(0, 0, 0, 0);
	}
}
/*
 void renderVolumeLight( Image<uchar3> out, const Volume & volume, const Matrix4 view, const float nearPlane, const float farPlane, const float largestep, const float3 light, const float3 ambient ){
 dim3 block(16,16);
 raycastLight<<<divup(out.size, block), block>>>( out,  volume, view, nearPlane, farPlane, volume.dim.x/volume.size.x, largestep, light, ambient );
 }
 */

/*
 void renderInput( Image<float3> pos3D, Image<float3> normal, Image<float> depth, const Volume volume, const Matrix4 view, const float nearPlane, const float farPlane, const float step, const float largestep){
 dim3 block(16,16);
 raycastInput<<<divup(pos3D.size, block), block>>>(pos3D, normal, depth, volume, view, nearPlane, farPlane, step, largestep);
 }
 */

__global__ void initVolumeKernel(Volume volume, const float2 val) {
	uint3 pos = make_uint3(thr2pos2());
	for (pos.z = 0; pos.z < volume.size.z; ++pos.z)
		volume.set(pos, val);
}

__global__ void raycastKernel(Image<float3> pos3D, Image<float3> normal,
		const Volume volume, const Matrix4 view, const float nearPlane,
		const float farPlane, const float step, const float largestep) {
	const uint2 pos = thr2pos2();

	const float4 hit = raycast(volume, pos, view, nearPlane, farPlane, step,
			largestep);
	if (hit.w > 0) {
		pos3D[pos] = make_float3(hit);
		float3 surfNorm = volume.grad(make_float3(hit));
		if (length(surfNorm) == 0) {
			normal[pos].x = INVALID;
		} else {
			normal[pos] = normalize(surfNorm);
		}
	} else {
		pos3D[pos] = make_float3(0);
		normal[pos] = make_float3(INVALID, 0, 0);
	}
}

__forceinline__ __device__ float sq(const float x) {
	return x * x;
}

__global__ void integrateKernel(Volume vol, const Image<float> depth,
		const Matrix4 invTrack, const Matrix4 K, const float mu,
		const float maxweight) {
	uint3 pix = make_uint3(thr2pos2());
	float3 pos = invTrack * vol.pos(pix);
	float3 cameraX = K * pos;
	const float3 delta = rotate(invTrack,
			make_float3(0, 0, vol.dim.z / vol.size.z));
	const float3 cameraDelta = rotate(K, delta);

	for (pix.z = 0; pix.z < vol.size.z; ++pix.z, pos += delta, cameraX +=
			cameraDelta) {
		if (pos.z < 0.0001f) // some near plane constraint
			continue;
		const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
				cameraX.y / cameraX.z + 0.5f);
		if (pixel.x < 0 || pixel.x > depth.size.x - 1 || pixel.y < 0
				|| pixel.y > depth.size.y - 1)
			continue;
		const uint2 px = make_uint2(pixel.x, pixel.y);
		if (depth[px] == 0)
			continue;
		const float diff = (depth[px] - cameraX.z)
				* sqrt(1 + sq(pos.x / pos.z) + sq(pos.y / pos.z));
		if (diff > -mu) {
			const float sdf = fminf(1.f, diff / mu);
			float2 data = vol[pix];
			data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f, 1.f);
			data.y = fminf(data.y + 1, maxweight);
			vol.set(pix, data);
		}
	}
}

__global__ void depth2vertexKernel(Image<float3> vertex,
		const Image<float> depth, const Matrix4 invK) {
	const uint2 pixel = thr2pos2();
	if (pixel.x >= depth.size.x || pixel.y >= depth.size.y)
		return;

	if (depth[pixel] > 0) {
		vertex[pixel] = depth[pixel]
				* (rotate(invK, make_float3(pixel.x, pixel.y, 1.f)));
	} else {
		vertex[pixel] = make_float3(0);
	}
}

__global__ void vertex2normalKernel(Image<float3> normal,
		const Image<float3> vertex) {
	const uint2 pixel = thr2pos2();
	if (pixel.x >= vertex.size.x || pixel.y >= vertex.size.y)
		return;

	const float3 left = vertex[make_uint2(max(int(pixel.x) - 1, 0), pixel.y)];
	const float3 right = vertex[make_uint2(min(pixel.x + 1, vertex.size.x - 1),
			pixel.y)];
	const float3 up = vertex[make_uint2(pixel.x, max(int(pixel.y) - 1, 0))];
	const float3 down = vertex[make_uint2(pixel.x,
			min(pixel.y + 1, vertex.size.y - 1))];

	if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
		normal[pixel].x = INVALID;
		return;
	}

	const float3 dxv = right - left;
	const float3 dyv = down - up;
	normal[pixel] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
}

template <int HALFSAMPLE>
__global__ void mm2metersKernel( Image<float> depth, const Image<ushort> in ) {
	const uint2 pixel = thr2pos2();
	depth[pixel] = in[pixel * (HALFSAMPLE+1)] / 1000.0f;
}

//column pass using coalesced global memory reads
__global__ void bilateralFilterKernel(Image<float> out, const Image<float> in,
		const Image<float> gaussian, const float e_d, const int r) {
	const uint2 pos = thr2pos2();

	if (in[pos] == 0) {
		out[pos] = 0;
		return;
	}

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[pos];

	for (int i = -r; i <= r; ++i) {
		for (int j = -r; j <= r; ++j) {
			const float curPix = in[make_uint2(
					clamp(pos.x + i, 0u, in.size.x - 1),
					clamp(pos.y + j, 0u, in.size.y - 1))];
			if (curPix > 0) {
				const float mod = sq(curPix - center);
				const float factor = gaussian[make_uint2(i + r, 0)]
						* gaussian[make_uint2(j + r, 0)]
						* __expf(-mod / (2 * e_d * e_d));
				t += factor * curPix;
				sum += factor;
			}
		}
	}
	out[pos] = t / sum;
}

// filter and halfsample
__global__ void halfSampleRobustImageKernel(Image<float> out,
		const Image<float> in, const float e_d, const int r) {
	const uint2 pixel = thr2pos2();
	const uint2 centerPixel = 2 * pixel;

	if (pixel.x >= out.size.x || pixel.y >= out.size.y)
		return;

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[centerPixel];
	for (int i = -r + 1; i <= r; ++i) {
		for (int j = -r + 1; j <= r; ++j) {
			float current = in[make_uint2(
					clamp(make_int2(centerPixel.x + j, centerPixel.y + i),
							make_int2(0),
							make_int2(in.size.x - 1, in.size.y - 1)))]; // TODO simplify this!
			if (fabsf(current - center) < e_d) {
				sum += 1.0f;
				t += current;
			}
		}
	}
	out[pixel] = t / sum;
}

__global__ void generate_gaussian(Image<float> out, float delta, int radius) {
	int x = threadIdx.x - radius;
	out[make_uint2(threadIdx.x, 0)] = __expf(-(x * x) / (2 * delta * delta));
}

__global__ void trackKernel(Image<TrackData> output,
		const Image<float3> inVertex, const Image<float3> inNormal,
		const Image<float3> refVertex, const Image<float3> refNormal,
		const Matrix4 Ttrack, const Matrix4 view, const float dist_threshold,
		const float normal_threshold) {
	const uint2 pixel = thr2pos2();
	if (pixel.x >= inVertex.size.x || pixel.y >= inVertex.size.y)
		return;

	TrackData & row = output[pixel];

	if (inNormal[pixel].x == INVALID) {
		row.result = -1;
		return;
	}

	const float3 projectedVertex = Ttrack * inVertex[pixel];
	const float3 projectedPos = view * projectedVertex;
	const float2 projPixel = make_float2(projectedPos.x / projectedPos.z + 0.5f,
			projectedPos.y / projectedPos.z + 0.5f);

	if (projPixel.x < 0 || projPixel.x > refVertex.size.x - 1 || projPixel.y < 0
			|| projPixel.y > refVertex.size.y - 1) {
		row.result = -2;
		return;
	}

	const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
	const float3 referenceNormal = refNormal[refPixel];

	if (referenceNormal.x == INVALID) {
		row.result = -3;
		return;
	}

	const float3 diff = refVertex[refPixel] - projectedVertex;
	const float3 projectedNormal = rotate(Ttrack, inNormal[pixel]);

	if (length(diff) > dist_threshold) {
		row.result = -4;
		return;
	}
	if (dot(projectedNormal, referenceNormal) < normal_threshold) {
		row.result = -5;
		return;
	}

	row.result = 1;
	row.error = dot(referenceNormal, diff);
	((float3 *) row.J)[0] = referenceNormal;
	((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
}

__global__ void reduceKernel(float * out, const Image<TrackData> J,
		const uint2 size) {
	__shared__
	float S[112][32]; // this is for the final accumulation
	const uint sline = threadIdx.x;

	float sums[32];
	float * jtj = sums + 7;
	float * info = sums + 28;

	for (uint i = 0; i < 32; ++i)
		sums[i] = 0;

	for (uint y = blockIdx.x; y < size.y; y += gridDim.x) {
		for (uint x = sline; x < size.x; x += blockDim.x) {
			const TrackData & row = J[make_uint2(x, y)];
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
				sums[i + 1] += row.error * row.J[i];

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

	for (int i = 0; i < 32; ++i) // copy over to shared memory
		S[sline][i] = sums[i];

	__syncthreads();            // wait for everyone to finish

	if (sline < 32) { // sum up columns and copy to global memory in the final 32 threads
		for (unsigned i = 1; i < blockDim.x; ++i)
			S[0][sline] += S[i][sline];
		out[sline + blockIdx.x * 32] = S[0][sline];
	}
}

Matrix4 operator*(const Matrix4 & A, const Matrix4 & B) {
	Matrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = TooN::wrapMatrix<4, 4>(&A.data[0].x)
			* TooN::wrapMatrix<4, 4>(&B.data[0].x);
	return R;
}

template<typename P>
inline Matrix4 toMatrix4(const TooN::SE3<P> & p) {
	const TooN::Matrix<4, 4, float> I = TooN::Identity;
	Matrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = p * I;
	return R;
}

template<typename P>
inline sMatrix4 tosMatrix4(const TooN::SE3<P> & p) {
	const TooN::Matrix<4, 4, float> I = TooN::Identity;
	sMatrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = p * I;
	return R;
}

Matrix4 inverse(const Matrix4 & A) {
	static TooN::Matrix<4, 4, float> I = TooN::Identity;
	TooN::Matrix<4, 4, float> temp = TooN::wrapMatrix<4, 4>(&A.data[0].x);
	Matrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = TooN::gaussian_elimination(temp, I);
	return R;
}

std::ostream & operator<<(std::ostream & out, const Matrix4 & m) {
	for (unsigned i = 0; i < 4; ++i)
		out << m.data[i].x << "  " << m.data[i].y << "  " << m.data[i].z << "  "
				<< m.data[i].w << "\n";
	return out;
}

template<typename P, typename A>
TooN::Matrix<6> makeJTJ(const TooN::Vector<21, P, A> & v) {
	TooN::Matrix<6> C = TooN::Zeros;
	C[0] = v.template slice<0, 6>();
	C[1].template slice<1, 5>() = v.template slice<6, 5>();
	C[2].template slice<2, 4>() = v.template slice<11, 4>();
	C[3].template slice<3, 3>() = v.template slice<15, 3>();
	C[4].template slice<4, 2>() = v.template slice<18, 2>();
	C[5][5] = v[20];

	for (int r = 1; r < 6; ++r)
		for (int c = 0; c < r; ++c)
			C[r][c] = C[c][r];

	return C;
}

template<typename T, typename A>
TooN::Vector<6> solve(const TooN::Vector<27, T, A> & vals) {
	const TooN::Vector<6> b = vals.template slice<0, 6>();
	const TooN::Matrix<6> C = makeJTJ(vals.template slice<6, 21>());

	TooN::GR_SVD<6, 6> svd(C);
	return svd.backsub(b, 1e6);
}

int printCUDAError() {
	cudaError_t error = cudaGetLastError();
	if (error)
		std::cout << cudaGetErrorString(error) << std::endl;
	return error;
}

/// OBJ ///

class Kfusion {
private:
	__device_builtin__uint2 computationSize;
	float step;
	sMatrix4 pose;
	sMatrix4 *viewPose;
	__device_builtin__float3 volumeDimensions;
	__device_builtin__uint3 volumeResolution;
	std::vector<int> iterations;
public:
	Kfusion(__device_builtin__uint2 inputSize,
			__device_builtin__uint3 volumeResolution,
			__device_builtin__float3 volumeDimensions,
			__device_builtin__float3 initPose, std::vector<int> & pyramid);

	void languageSpecificConstructor();
	~Kfusion();

	void reset();

	inline void computeFrame(const ushort * inputDepth,
			const __device_builtin__uint2 inputSize, __device_builtin__float4 k,
			uint integration_rate, uint tracking_rate, float icp_threshold,
			float mu, const uint frame) {
		preprocessing(inputDepth, inputSize);
		tracking(k, icp_threshold, tracking_rate, frame);
		integration(k, integration_rate, mu, frame);
		raycasting(k, mu, frame);
	}

	bool preprocessing(const ushort * inputDepth,
			const __device_builtin__uint2 inputSize);
	bool tracking(__device_builtin__float4 k, float icp_threshold,
			uint tracking_rate, uint frame);
	bool raycasting(__device_builtin__float4 k, float mu, uint frame);
	bool integration(__device_builtin__float4 k, uint integration_rate,
			float mu, uint frame);

	void dumpVolume(std::string filename);
	void renderVolume(__device_builtin__uchar4 * out,
			const __device_builtin__uint2 outputSize, int, int,
			__device_builtin__float4 k, float largestep);
	void renderTrack(__device_builtin__uchar4 * out,
			const __device_builtin__uint2 outputSize);
	void renderDepth(__device_builtin__uchar4 * out,
			__device_builtin__uint2 outputSize);
	sMatrix4 getPose() {
		return pose;
	}
	void setViewPose(sMatrix4 *value = NULL) {
		if (value == NULL)
			viewPose = &pose;
		else
			viewPose = value;
	}
	sMatrix4 *getViewPose() {
		return (viewPose);
	}
	__device_builtin__float3 getModelDimensions() {
		return (volumeDimensions);
	}
	__device_builtin__uint3 getModelResolution() {
		return (volumeResolution);
	}
	__device_builtin__uint2 getComputationResolution() {
		return (computationSize);
	}

};

// NVIDIA SPECIFIC

dim3 imageBlock = dim3(32, 16);
dim3 raycastBlock = dim3(32, 8);

sMatrix4 oldPose;
sMatrix4 raycastPose;

Volume volume;
Image<TrackData, Device> reduction;
Image<float3, Device> vertex, normal;

std::vector<Image<float3, Device> > inputVertex, inputNormal;
std::vector<Image<float, Device> > scaledDepth;

Image<float, Device> rawDepth;
Image<float, HostDevice> output;

Image<float, Device> gaussian;

Image<uchar3, HostDevice> lightScene, texModel;
Image<uchar4, HostDevice> lightModel, trackModel, depthModel;


static bool firstAcquire = true;

void Kfusion::languageSpecificConstructor() {
	if (getenv("KERNEL_TIMINGS"))
		print_kernel_timing = true;
	if (firstAcquire)
		cudaSetDeviceFlags (cudaDeviceMapHost);
	//size_t freeMem, totalMem;
	//cudaMemGetInfo(&freeMem, &totalMem);
	//std::cerr<< "Total mem: " << totalMem/(1024*1024) << " freeMem: " << freeMem/(1024*1024) << "\n";
	uint3 vr = make_uint3(volumeResolution.x, volumeResolution.y,
			volumeResolution.z);
	float3 vd = make_float3(volumeDimensions.x, volumeDimensions.y,
			volumeDimensions.z);
	volume.init(vr, vd);
	uint2 cs = make_uint2(this->computationSize.x, this->computationSize.y);
	reduction.alloc(cs);
	vertex.alloc(cs);
	normal.alloc(cs);
	rawDepth.alloc(cs);
	scaledDepth.resize(iterations.size());
	inputVertex.resize(iterations.size());
	inputNormal.resize(iterations.size());

	for (int i = 0; i < iterations.size(); ++i) {
		scaledDepth[i].alloc(cs >> i);
		inputVertex[i].alloc(cs >> i);
		inputNormal[i].alloc(cs >> i);
	}

	gaussian.alloc(make_uint2(radius * 2 + 1, 1));
	output.alloc(make_uint2(32, 8));
	//generate gaussian array
	generate_gaussian<<< 1,gaussian.size.x>>>(gaussian, delta, radius);
	dim3 block(32, 16);
	dim3 grid = divup(dim3(volume.size.x, volume.size.y), block);
	TICK("initVolume");
	initVolumeKernel<<<grid, block>>>(volume, make_float2(1.0f, 0.0f));
	TOCK();

	// input buffers
	depthModel.alloc(cs);
	// render buffers
	trackModel.alloc(cs);
	//lightModel is used to render the volume, let it be upt0 640x480 so that we don't have to render at inputSize/computeRatio
	// which can be too small to be useful   
	lightModel.alloc(
			make_uint2(
					this->computationSize.x < 640 ?
							640 : this->computationSize.x,
					this->computationSize.y < 480 ?
							480 : this->computationSize.y));
	if (printCUDAError()) {
		cudaDeviceReset();
		exit(1);
	}
	firstAcquire = false;

	memset(depthModel.data(), 0,
			depthModel.size.x * depthModel.size.y * sizeof(uint16_t));
}

Kfusion::~Kfusion() {
	cudaThreadSynchronize();
	volume.release();
	printCUDAError();
}
void Kfusion::reset() {

	dim3 block(32, 16);
	dim3 grid = divup(dim3(volume.size.x, volume.size.y), block);
initVolumeKernel<<<grid, block>>>(volume, make_float2(1.0f, 0.0f));

}
void init() {
}
void clean() {
}

bool updatePoseKernel(sMatrix4 & pose, const float * output,
	float icp_threshold) {

// Update the pose regarding the tracking result
TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
TooN::Vector<6> x = solve(values[0].slice<1, 27>());
TooN::SE3<> delta(x);
Matrix4 delta4 = toMatrix4(delta) * Matrix4(&pose);

pose.data[0].x = delta4.data[0].x;
pose.data[0].y = delta4.data[0].y;
pose.data[0].z = delta4.data[0].z;
pose.data[0].w = delta4.data[0].w;
pose.data[1].x = delta4.data[1].x;
pose.data[1].y = delta4.data[1].y;
pose.data[1].z = delta4.data[1].z;
pose.data[1].w = delta4.data[1].w;
pose.data[2].x = delta4.data[2].x;
pose.data[2].y = delta4.data[2].y;
pose.data[2].z = delta4.data[2].z;
pose.data[2].w = delta4.data[2].w;
pose.data[3].x = delta4.data[3].x;
pose.data[3].y = delta4.data[3].y;
pose.data[3].z = delta4.data[3].z;
pose.data[3].w = delta4.data[3].w;

// Return validity test result of the tracking
if (norm(x) < icp_threshold)
	return true;
return false;

}

bool checkPoseKernel(sMatrix4 & pose, sMatrix4 oldPose, const float * output,
	__device_builtin__uint2 imageSize, float track_threshold) {

// Check the tracking result, and go back to the previous camera position if necessary

TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2)
		|| (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold)) {
	pose = oldPose;
	return false;
} else {
	return true;
}

}

bool Kfusion::preprocessing(const ushort * inputDepth, const __device_builtin__uint2 inputSize) {
    uint2 s = make_uint2(inputSize.x, inputSize.y);

    //Image<uint16_t, Ref> myDepthImage(s,(void*)inputDepth);
    Image<uint16_t, HostDevice> myDepthImage(s);
    cudaMemcpy(myDepthImage.data(), inputDepth, s.x * s.y * sizeof(ushort),
	    cudaMemcpyHostToHost);
    TICK("mm2meters");
    if(computationSize.x == myDepthImage.size.x)
	mm2metersKernel<0><<<divup(rawDepth.size, imageBlock), imageBlock>>>(rawDepth, myDepthImage);
    else if(computationSize.x == myDepthImage.size.x / 2 )
	mm2metersKernel<1><<<divup(rawDepth.size, imageBlock), imageBlock>>>(rawDepth, myDepthImage);
    else if(computationSize.x == myDepthImage.size.x / 4 )
	mm2metersKernel<3><<<divup(rawDepth.size, imageBlock), imageBlock>>>(rawDepth, myDepthImage);
    else if(computationSize.x == myDepthImage.size.x / 8 )
	mm2metersKernel<7><<<divup(rawDepth.size, imageBlock), imageBlock>>>(rawDepth, myDepthImage);
    else
	assert(false);
    TOCK();

    // filter the input depth map
    dim3 grid = divup(make_uint2(computationSize.x, computationSize.y), imageBlock);
    TICK("bilateral_filter");
    bilateralFilterKernel<<<grid, imageBlock>>>(scaledDepth[0], rawDepth, gaussian, e_delta, radius);
    TOCK();
    
    return true;
}

bool Kfusion::tracking(__device_builtin__float4 sk, float icp_threshold, uint tracking_rate, uint frame) {

if (frame % tracking_rate != 0)
	return false;

float4 k = make_float4(sk.x, sk.y, sk.z, sk.w);
const Matrix4 invK = getInverseCameraMatrix(k);

vector<dim3> grids;
for (int i = 0; i < iterations.size(); ++i)
	grids.push_back(divup(make_uint2(computationSize.x, computationSize.y) >> i, imageBlock));

// half sample the input depth maps into the pyramid levels
for (int i = 1; i < iterations.size(); ++i) {
	TICK("halfSampleRobust");
	halfSampleRobustImageKernel<<<grids[i], imageBlock>>>(scaledDepth[i], scaledDepth[i-1], e_delta * 3, 1);
	TOCK();
}
// prepare the 3D information from the input depth maps
for (int i = 0; i < iterations.size(); ++i) {
	TICK("depth2vertex");
	depth2vertexKernel<<<grids[i], imageBlock>>>( inputVertex[i], scaledDepth[i], getInverseCameraMatrix(k / float(1 << i))); // inverse camera matrix depends on level
	TOCK();
	TICK("vertex2normal");
	vertex2normalKernel<<<grids[i], imageBlock>>>( inputNormal[i], inputVertex[i] );
	TOCK();
}

oldPose = pose;
const Matrix4 projectReference = getCameraMatrix(k)
		* inverse(Matrix4(&raycastPose));

for (int level = iterations.size() - 1; level >= 0; --level) {
	for (int i = 0; i < iterations[level]; ++i) {
		TICK("track");
		trackKernel<<<grids[level], imageBlock>>>( reduction, inputVertex[level], inputNormal[level], vertex, normal, Matrix4( & pose ), projectReference, dist_threshold, normal_threshold);
		TOCK();
		TICK("reduce");
		reduceKernel<<<8, 112>>>( output.getDeviceImage().data(), reduction, inputVertex[level].size ); // compute the linear system to solve
		TOCK();
		cudaDeviceSynchronize();// important due to async nature of kernel call

		TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(output.data());
		for(int j = 1; j < 8; ++j)
		values[0] += values[j];
		if (updatePoseKernel(pose, output.data(), icp_threshold))
		break;
	}
}
return checkPoseKernel(pose, oldPose, output.data(), computationSize,
		track_threshold);

}

bool Kfusion::raycasting(__device_builtin__float4 sk, float mu, uint frame) {

bool doRaycast = false;
float4 k = make_float4(sk.x, sk.y, sk.z, sk.w);

if (frame > 2) {
	raycastPose = pose;
	TICK("raycast");
	raycastKernel<<<divup(make_uint2(computationSize.x,computationSize.y), raycastBlock), raycastBlock>>>(vertex, normal, volume, Matrix4(&raycastPose) * getInverseCameraMatrix(k), nearPlane, farPlane, step, 0.75f * mu);
	TOCK();
}

return doRaycast;

}

bool Kfusion::integration(__device_builtin__float4 sk, uint integration_rate,
	float mu, uint frame) {

float4 k = make_float4(sk.x, sk.y, sk.z, sk.w);

bool doIntegrate = checkPoseKernel(pose, oldPose, output.data(),
		computationSize, track_threshold);
if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
	TICK("integrate");
	integrateKernel<<<divup(dim3(volume.size.x, volume.size.y), imageBlock), imageBlock>>>( volume, rawDepth, inverse(Matrix4(&pose)), getCameraMatrix(k), mu, maxweight );
	TOCK();
	doIntegrate = true;
} else {
	doIntegrate = false;
}

return doIntegrate;

}

void Kfusion::dumpVolume(std::string filename) {
std::ofstream fDumpFile;

if (filename.compare("") == 0) {
	return;
}

cout << "Dumping the volumetric representation on file: " << filename << endl;
fDumpFile.open(filename.c_str(), ios::out | ios::binary);
if (fDumpFile == NULL) {
	cout << "Error opening file: " << filename << endl;
	exit(1);
}

// Retrieve the volumetric representation data from the GPU
short2 *hostData = (short2 *) malloc(
		volume.size.x * volume.size.y * volume.size.z * sizeof(short2));
if (cudaMemcpy(hostData, volume.data,
		volume.size.x * volume.size.y * volume.size.z * sizeof(short2),
		cudaMemcpyDeviceToHost) != cudaSuccess) {
	cout << "Error reading volumetric representation data from the GPU. "
			<< endl;
	exit(1);
}

// Dump on file without the y component of the short2 variable
for (int i = 0; i < volume.size.x * volume.size.y * volume.size.z; i++) {
	fDumpFile.write((char *) (hostData + i), sizeof(short));
}

fDumpFile.close();

if (hostData) {
	free(hostData);
	hostData = NULL;
}
}

void Kfusion::renderVolume(__device_builtin__uchar4 * out,
	__device_builtin__uint2 outputSize, int frame, int rate,
	__device_builtin__float4 k, float largestep) {
if (frame % rate != 0)
	return;
TICK("renderVolume");
dim3 block(16,16);
//So We initially create a fixed size image (possibly 640x480) but we need to set up the dimensions for
//the image as the CUDA accesss by row, column 
if((outputSize.x != lightModel.size.x) || (outputSize.y != lightModel.size.y)) {
	lightModel.size.x = outputSize.x;
	lightModel.size.y = outputSize.y;
}
renderVolumeKernel<<<divup(lightModel.getDeviceImage().size, block), block>>>( lightModel.getDeviceImage(),volume,Matrix4(this->viewPose) * getInverseCameraMatrix(make_float4(k.x,k.y,k.z,k.w)) , nearPlane, farPlane, volume.dim.x/volume.size.x, largestep,light, ambient );
TOCK();
cudaMemcpy(out, lightModel.getDeviceImage().data(),
		outputSize.x * outputSize.y * sizeof(uchar4),
		cudaMemcpyDeviceToHost);
}

void Kfusion::renderTrack(__device_builtin__uchar4 * out,
	__device_builtin__uint2 outputSize) {
TICK("renderTrack");
dim3 block(32,16);
renderTrackKernel<<<divup(trackModel.getDeviceImage().size, block), block>>>( trackModel.getDeviceImage(), reduction );
TOCK();
cudaMemcpy(out, trackModel.getDeviceImage().data(), outputSize.x * outputSize.y * sizeof(uchar4), cudaMemcpyDeviceToHost);
}

void Kfusion::renderDepth(__device_builtin__uchar4 * out,
		__device_builtin__uint2 outputSize) {
	TICK("renderDepthKernel");
	dim3 block(32,16);
	renderDepthKernel<<<divup(depthModel.getDeviceImage().size, block), block>>>( depthModel.getDeviceImage(), rawDepth, nearPlane, farPlane );
	TOCK();
	cudaMemcpy(out, depthModel.getDeviceImage().data(), outputSize.x * outputSize.y * sizeof(uchar4), cudaMemcpyDeviceToHost);
}

void synchroniseDevices() {
	cudaDeviceSynchronize();
}
