/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef _SYCL_KERNELS_
#define _SYCL_KERNELS_

#include <cstdlib>
#include <sycl/commons.h>

////////////////////////// COMPUTATION KERNELS PROTOTYPES //////////////////////

bool updatePoseKernel(Matrix4 & pose, const float * output,
		float icp_threshold);

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output,
		uint2 imageSize, float track_threshold);

void renderNormaKernell(uchar3* out, const float3* normal, uint2 normalSize);

////////////////////////// MULTI-KERNELS PROTOTYPES //////////////////////
template <typename T>
void computeFrame(Volume<T> & integration, float3 * vertex, float3 * normal,
		TrackData * trackingResult, Matrix4 & pose, const float * inputDepth,
		const uint2 inputSize, const float * gaussian,
		const std::vector<int> iterations, float4 k, const uint frame);

void init();

void clean();

/// OBJ ///

class Kfusion {
private:
	uint2 computationSize;
	float step;
	Matrix4 pose;
	Matrix4 *viewPose;
	float3 volumeDimensions;
	uint3 volumeResolution;
	std::vector<int> iterations;
	bool _tracked;
	bool _integrated;
	float3 _initPose;

	void raycast(uint frame, const float4& k, float mu);

public:
	Kfusion(uint2 inputSize, uint3 volumeResolution, float3 volumeDimensions,
			float3 initPose, std::vector<int> & pyramid) :
			computationSize(make_uint2(inputSize.x(), inputSize.y())) {

		this->_initPose = initPose;
		this->volumeDimensions = volumeDimensions;
		this->volumeResolution = volumeResolution;
		pose = toMatrix4(
				TooN::SE3<float>(
						TooN::makeVector(initPose.x(), initPose.y(), initPose.z(), 0,
								0, 0)));
		this->iterations.clear();
		for (std::vector<int>::iterator it = pyramid.begin();
				it != pyramid.end(); it++) {
			this->iterations.push_back(*it);
		}

		step = min(volumeDimensions) / max(volumeResolution);
		viewPose = &pose;
		this->languageSpecificConstructor();
	}
//Allow a kfusion object to be created with a pose which include orientation as well as position
	Kfusion(uint2 inputSize, uint3 volumeResolution, float3 volumeDimensions,
			Matrix4 initPose, std::vector<int> & pyramid) :
			computationSize(make_uint2(inputSize.x(), inputSize.y())) {
		this->_initPose = getPosition();
		this->volumeDimensions = volumeDimensions;
		this->volumeResolution = volumeResolution;
		pose = initPose;

		this->iterations.clear();
		for (std::vector<int>::iterator it = pyramid.begin();
				it != pyramid.end(); it++) {
			this->iterations.push_back(*it);
		}

		step = min(volumeDimensions) / max(volumeResolution);
		viewPose = &pose;
		this->languageSpecificConstructor();
	}

	void languageSpecificConstructor();
	~Kfusion();

	void reset();
	bool getTracked() {
		return (_tracked);
	}
	bool getIntegrated() {
		return (_integrated);
	}
	float3 getPosition() {
		//std::cerr << "InitPose =" << _initPose.x() << "," << _initPose.y()  <<"," << _initPose.z() << "    ";
		//std::cerr << "pose =" << pose.data[0].w() << "," << pose.data[1].w()  <<"," << pose.data[2].w() << "    ";
		float xt = pose.data[0].w() - _initPose.x();
		float yt = pose.data[1].w() - _initPose.y();
		float zt = pose.data[2].w() - _initPose.z();
		return (make_float3(xt, yt, zt));
	}
	inline void computeFrame(const ushort * inputDepth, const uint2 inputSize,
			float4 k, uint integration_rate, uint tracking_rate,
			float icp_threshold, float mu, const uint frame) {
		preprocessing(inputDepth, inputSize);
		_tracked = tracking(k, icp_threshold, tracking_rate, frame);
		_integrated = integration(k, integration_rate, mu, frame);
		raycasting(k, mu, frame);
	}

	bool preprocessing(const ushort * inputDepth, const uint2 inputSize);
	bool tracking(float4 k, float icp_threshold, uint tracking_rate,
			uint frame);
	bool raycasting(float4 k, float mu, uint frame);
	bool integration(float4 k, uint integration_rate, float mu, uint frame);

	void dumpVolume(std::string filename);
	void renderVolume(uchar4 *out, const uint2 outputSize, const int frame,
                    const int rate, const float4 k, const float mu);
	void renderTrack(uchar4 *out, const uint2 outputSize);
	void renderDepth(uchar4 *out, const uint2 outputSize);
	Matrix4 getPose() {
		return pose;
	}
	void setViewPose(Matrix4 *value = NULL) {
		if (value == NULL)
			viewPose = &pose;
		else
			viewPose = value;
	}
	Matrix4 *getViewPose() {
		return (viewPose);
	}
	float3 getModelDimensions() {
		return (volumeDimensions);
	}
	uint3 getModelResolution() {
		return (volumeResolution);
	}
	uint2 getComputationResolution() {
		return (computationSize);
	}

};

void synchroniseDevices(); // Synchronise CPU and GPU

#endif  // _SYCL_KERNELS_
