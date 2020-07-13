/*

 Copyright (c) 2015-2019 Paul Keir, University of the West of Scotland.
 This code is licensed under the MIT License.

 */
#include <CL/sycl.hpp>
#include <kernels.h>

// input once
float *gaussian;

// inter-frame
Volume<short2 *> volume;

// intra-frame
Matrix4 oldPose, raycastPose;

// sycl specific
cl::sycl::queue q;

buffer<float,1>      *ocl_gaussian             = NULL;

buffer<float3,1>     *ocl_vertex               = NULL;
buffer<float3,1>     *ocl_normal               = NULL;
buffer<short2,1>     *ocl_volume_data          = NULL;

buffer<TrackData,1>  *ocl_trackingResult       = NULL;
buffer<float,1>      *ocl_FloatDepth           = NULL;
buffer<float,1>     **ocl_ScaledDepth          = NULL;
buffer<float3,1>    **ocl_inputVertex          = NULL;
buffer<float3,1>    **ocl_inputNormal          = NULL;
float                *reduceOutputBuffer       = NULL;

// reduction parameters
static const size_t size_of_group    = 64;
static const size_t number_of_groups = 8;
const float inv_32766 = 0.00003051944088f;

namespace kernels {
  class initVolumeKernel;
  class mm2metersKernel;
  class bilateralFilterKernel;
  class halfSampleRobustImageKernel;
  class vertex2normalKernel;
  class depth2vertexKernel;
  class trackKernel;
  class reduceKernel;
  class integrateKernel;
  class raycastKernel;
  class renderDepthKernel;
  class renderTrackKernel;
  class renderVolumeKernel;
}

void Kfusion::languageSpecificConstructor() {

  const uint csize = computationSize.x() * computationSize.y();
  
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
  ocl_trackingResult = new buffer<TrackData,1>(range<1>{csize});

	reduceOutputBuffer = (float*) malloc(number_of_groups * 32 * sizeof(float));

	// ********* BEGIN : Generate the gaussian *************
	size_t gaussianS = radius * 2 + 1;
	gaussian = (float*) calloc(gaussianS * sizeof(float), 1);
	for (unsigned int i = 0; i < gaussianS; i++) {
		int x = i - 2;
		gaussian[i] = expf(-(x * x) / (2 * delta * delta));
	}

  ocl_gaussian = new buffer<float,1>(gaussian, range<1>{gaussianS});
	// ********* END : Generate the gaussian *************

  const uint vsize = volumeResolution.x() *
                     volumeResolution.y() * volumeResolution.z();
  ocl_volume_data = new buffer<short2>(range<1>{vsize});

	volume.init(volumeResolution, volumeDimensions);
	reset();
}

Kfusion::~Kfusion() {

	if (reduceOutputBuffer) {
    free(reduceOutputBuffer);
    reduceOutputBuffer = NULL;
  }

	if (ocl_FloatDepth) {
    delete ocl_FloatDepth;
    ocl_FloatDepth = NULL;
  }

	for (unsigned int i = 0; i < iterations.size(); ++i)
  {
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
	if (ocl_gaussian) {
		delete ocl_gaussian;
		ocl_gaussian = NULL;
  }
	if (ocl_volume_data) {
		delete ocl_volume_data;
		ocl_volume_data = NULL;
  }
	if (ocl_trackingResult) {
		delete ocl_trackingResult;
		ocl_trackingResult = NULL;
  }

	free(gaussian);
	volume.release();
}

void Kfusion::reset() {
  const uint3 &vr = volumeResolution; // declared in kernels.h
  const auto r = range<3>{vr.x(),vr.y(),vr.z()};

  q.submit([&](handler &cgh) {

    using namespace cl::sycl::access;
    const auto data = ocl_volume_data->get_access<mode::write>(cgh);

    cgh.parallel_for<kernels::initVolumeKernel>(r, [=](const item<3> ix) {
      uint3 size{ix.get_range()[0], ix.get_range()[1], ix.get_range()[2]};
      auto &d = data[ix[0] + ix[1] * size.x() + ix[2] * size.x() * size.y()];
      d = {32766.0f, 0.0f};
    });
  });
}

void init()  { } // stub
void clean() { } // stub

template <typename T>
inline void setVolume(Volume<T> v, const uint3 pos, const float2 d) {
  const short2 d2((float)d.x() * 32766.0f, d.y());
  v.data.get()[(uint)pos.x() +
               (uint)pos.y() * (uint)v.size.x() +
               (uint)pos.z() * (uint)v.size.x() * (uint)v.size.y()] = d2;
}

template <typename T>
inline float3 posVolume(const Volume<T> v, const uint3 p) {
	return float3{((uint)p.x() + 0.5f) * v.dim.x() / v.size.x(),
                ((uint)p.y() + 0.5f) * v.dim.y() / v.size.y(),
                ((uint)p.z() + 0.5f) * v.dim.z() / v.size.z()};
}

template <typename T>
inline float2 getVolume(const Volume<T> v, const uint3 pos) {
  const short2 d =
    v.data.get()[(uint)pos.x() +
                 (uint)pos.y() * (uint)v.size.x() +
                 (uint)pos.z() * (uint)v.size.x() * (uint)v.size.y()];
	return float2{(short)d.x() * inv_32766, d.y()};
}

bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold)
{
	bool res = false;
	// Update the pose regarding the tracking result
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	pose = toMatrix4(delta) * pose;

	// Return validity test result of the tracking
	if (norm(x) < icp_threshold)
		res = true;

	return res;
}

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output,
		uint2 imageSize, float track_threshold) {

	// Check the tracking result, and go back to the previous camera position if necessary

	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

	if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2)
			|| (values(0, 28) / (((uint)imageSize.x()) * ((uint)imageSize.y())) < track_threshold)) {
		pose = oldPose;
		return false;
	} else {
		return true;
	}
}

bool Kfusion::preprocessing(const uint16_t *inputDepth, const uint2 inSize)
{
	uint2 outSize = computationSize;

	// Check for unsupported conditions
	if (((uint)inSize.x() < (uint)outSize.x()) || ((uint)inSize.y() < (uint)outSize.y())) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if (((uint)inSize.x() % (uint)outSize.x() != 0) || ((uint)inSize.y() % (uint)outSize.y() != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if (((uint)inSize.x() / (uint)outSize.x() != (uint)inSize.y() / (uint)outSize.y())) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = (uint)inSize.x() / (uint)outSize.x();

  using namespace cl::sycl::access;
  const auto r = range<2>{outSize.x(),outSize.y()};
	buffer<uint16_t,1> idb(inputDepth, range<1>{(uint)inSize.x() * (uint)inSize.y()});

  q.submit([&](handler &cgh) {
    const auto depth = ocl_FloatDepth->get_access<mode::read_write>(cgh);
    const auto    in = idb.get_access<mode::read>(cgh);
    cgh.parallel_for<kernels::mm2metersKernel>(r, [=](const item<2> ix) {
      depth[ix[0] + (uint)outSize.x() * ix[1]] =
         in[ix[0] * ratio + (uint)inSize.x() * ix[1] * ratio] / 1000.0f;
    });
  });

  q.submit([&](handler &cgh) {
    const auto      out = ocl_ScaledDepth[0]->get_access<mode::read_write>(cgh);
    const auto       in = ocl_FloatDepth->get_access<mode::read>(cgh);
    const auto gaussian = ocl_gaussian->get_access<mode::read>(cgh);
    cgh.parallel_for<kernels::bilateralFilterKernel>(r, [=](const item<2> ix) {

      const float center = in[ix[0] + ix.get_range()[0] * ix[1]];

      if (center == 0) {
        out[ix[0] + ix.get_range()[0] * ix[1]] = 0;
        return;
      }

      float sum = 0.0f;
      float t   = 0.0f;
      for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
          // n.b. unsigned + signed is unsigned! Bug in OpenCL C version?
          const int px = ix[0] + i; const int sx = ix.get_range()[0] - 1;
          const int py = ix[1] + i; const int sy = ix.get_range()[1] - 1;
          const int   curPosx  = clamp(px,0,sx);
          const int   curPosy  = clamp(py,0,sy);
          const float curPix   = in[curPosx + curPosy * ix.get_range()[0]];
          if (curPix > 0) {
            const float mod    = sq(curPix - center);
            const float factor = gaussian[i + radius] * gaussian[j + radius] *
                                 cl::sycl::exp(-mod / (2 * e_delta * e_delta));
            t   += factor * curPix;
            sum += factor;
          } else {
            // std::cerr << "ERROR BILATERAL " << ix[0]+i << " " <<
            // ix[1]+j<< " " <<curPix<<" \n";
          }
        }
      } 
      out[ix[0] + ix.get_range()[0] * ix[1]] = t / sum;
    });
  });

	return true;
}

bool Kfusion::tracking(float4 k, float icp_threshold,
                       const uint tracking_rate, const uint frame)
{
  using namespace cl::sycl::access;
	if (frame % tracking_rate != 0)
		return false;

	// half sample the input depth maps into the pyramid levels
	for (unsigned int i = 1; i < iterations.size(); ++i) {
		uint2 outSize{(uint)computationSize.x() / (int) ::pow(2, i),
                  (uint)computationSize.y() / (int) ::pow(2, i)};

    const auto r   = range<2>{outSize.x(),outSize.y()};
		uint2 inSize{((uint)outSize.x())*2,((uint)outSize.y())*2}; // Seems redundant
    q.submit([&](handler &cgh) {
      const auto  in = ocl_ScaledDepth[i-1]->get_access<mode::read>(cgh);
      const auto out = ocl_ScaledDepth[i  ]->get_access<mode::read_write>(cgh);
      cgh.parallel_for<kernels::halfSampleRobustImageKernel>(
        r,[=](const item<2> ix
      ) {

        const uint2 centerPixel{2*ix[0], 2*ix[1]};

        float sum          = 0.0f;
        float t            = 0.0f;
        const int r        = 1;
        const float e_d    = e_delta * 3;
        const float center = in[(uint)centerPixel.x()+(uint)centerPixel.y()*(uint)inSize.x()];
        for (int i = -r + 1; i <= r; ++i) {
          for (int j = -r + 1; j <= r; ++j) {
            const int2 x{(uint)centerPixel.x()+j, (uint)centerPixel.y()+i};
            const int2 minval{0,0};
            const int2 maxval{(uint)inSize.x()-1, (uint)inSize.y()-1};
                  int2 from{clamp(x,minval,maxval)};
            float current = in[(int)from.x() + (int)from.y() * (uint)inSize.x()];
            if (cl::sycl::fabs(current - center) < e_d) {
              sum += 1.0f;
              t += current;
            }
          }
        }
        out[ix[0] + ix[1] * (uint)outSize.x()] = t / sum;
      });
    });
  }

	// prepare the 3D information from the input depth maps
	uint2 localimagesize = computationSize;
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));
    const range<2> r{localimagesize.x(),localimagesize.y()};

    q.submit([&](handler &cgh) {
      const auto vertex = ocl_inputVertex[i]->get_access<mode::read_write>(cgh);
      const auto depth  = ocl_ScaledDepth[i]->get_access<mode::read>(cgh);
      cgh.parallel_for<kernels::depth2vertexKernel>(r,[=](const item<2> ix) {
        float3 res{0,0,0};

        const float elem = depth[ix[0] + ix.get_range()[0] * ix[1]];
        if (elem > 0) {
          res = elem * rotate(invK, {ix[0], ix[1], 1});
        }

        vertex[ix[0] + ix.get_range()[0] * ix[1]] = res;
      });
    });

    q.submit([&](handler &cgh) {
      const auto normal = ocl_inputNormal[i]->get_access<mode::read_write>(cgh);
      const auto vertex = ocl_inputVertex[i]->get_access<mode::read>(cgh);
      cgh.parallel_for<kernels::vertex2normalKernel>(r,[=](const item<2> ix) {
        uint2  vleft{max((int)ix[0] - 1, 0),                          ix[1]};
        uint2 vright{min((int)ix[0] + 1, (int)ix.get_range()[0] - 1), ix[1]};
        uint2    vup{ix[0], max((int)ix[1] - 1, 0)};
        uint2  vdown{ix[0], min((int)ix[1] + 1, (int)ix.get_range()[1]-1)};

        const float3 left  = vertex[(uint)vleft.x()  + ix.get_range()[0] * (uint)vleft.y()];
        const float3 right = vertex[(uint)vright.x() + ix.get_range()[0] * (uint)vright.y()];
        const float3 up    = vertex[(uint)vup.x()    + ix.get_range()[0] * (uint)vup.y()];
        const float3 down  = vertex[(uint)vdown.x()  + ix.get_range()[0] * (uint)vdown.y()];

        if ((float)left.z() == 0.0f || (float)right.z() == 0.0f ||
            (float)  up.z() == 0.0f || (float) down.z() == 0.0f) {
          const float3 invalid3{KFUSION_INVALID,KFUSION_INVALID,KFUSION_INVALID};
          normal[ix[0] + ix.get_range()[0] * ix[1]] = invalid3;
          return;
        }

        const float3 dxv = right - left;
        const float3 dyv = down  - up;

        normal[ix[0] + ix.get_range()[0] * ix[1]] = normalize(cross(dyv,dxv));
      });
    });

    localimagesize = localimagesize / 2;
	}

	oldPose = pose;
	const Matrix4 view = getCameraMatrix(k) * inverse(raycastPose);
	const uint2 outputSize = computationSize;
	
  // iterations: a vector<int> set to {10,5,4} in Kfusion ctor (kernels.h)
  for (int level = iterations.size() - 1; level >= 0; --level) {	  
    const int csize_x = computationSize.x();
    const int csize_y = computationSize.y();
    const int pow2l = pow(2, level);
		uint2 localimagesize = make_uint2(csize_x / pow2l, csize_y / pow2l);

    for (int i = 0; i < iterations[level]; ++i) {  // i<4,i<5,i<10 
      range<2> r{localimagesize.x(),localimagesize.y()};
	
      q.submit([&](handler &cgh) {
        const Matrix4 pose = this->pose;
        const auto output=ocl_trackingResult->get_access<mode::read_write>(cgh);
        const auto inVertex=ocl_inputVertex[level]->get_access<mode::read>(cgh);
        const auto inNormal=ocl_inputNormal[level]->get_access<mode::read>(cgh);
        const auto refVertex=ocl_vertex->get_access<mode::read>(cgh);
        const auto refNormal=ocl_normal->get_access<mode::read>(cgh);
        cgh.parallel_for<kernels::trackKernel>(r,[=](const item<2> ix) {
          uint2 pixel{ix[0],ix[1]};
             
          TrackData &row = output[(uint)pixel.x() + (uint)outputSize.x() * (uint)pixel.y()];
         
          float3 inNormalPixel = inNormal[((uint)pixel.x()) + ix.get_range()[0] * ((uint)pixel.y())]; 
          if ((float)inNormalPixel.x() == KFUSION_INVALID) {
            row.result = -1;
            return;
          }  
          
          float3 inVertexPixel = inVertex[(uint)pixel.x() + ix.get_range()[0] * (uint)pixel.y()];
          const float3 projectedVertex = pose * inVertexPixel;
          const float3 projectedPos    = view * projectedVertex;
          const float2 projPixel{projectedPos.x() / projectedPos.z() + 0.5f,
                                 projectedPos.y() / projectedPos.z() + 0.5f};
          if ((float)projPixel.x() < 0.0f ||
              (float)projPixel.x() > (float)((uint)outputSize.x()-1) ||
              (float)projPixel.y() < 0.0f ||
              (float)projPixel.y() > (float)((uint)outputSize.y()-1)) {
            row.result = -2;
            return;
          }

          const uint2 refPixel{projPixel.x(), projPixel.y()};
          const float3 referenceNormal =
          refNormal[(uint)refPixel.x() + (uint)outputSize.x() * (uint)refPixel.y()];
          if ((float)referenceNormal.x() == KFUSION_INVALID) {
            row.result = -3;
            return;
          }

          const float3 diff = refVertex[(uint)refPixel.x() + (uint)outputSize.x() * (uint)refPixel.y()] - projectedVertex; 
          const float3 projectedNormal = rotate(pose, inNormalPixel);

          if (length(diff) > dist_threshold) {
            row.result = -4;
            return;
          }

          if (dot(projectedNormal,referenceNormal) < normal_threshold) {
            row.result = -5;
            return;
          }
          
          row.result = 1;
          row.error  = dot(referenceNormal, diff);
         
          const cl::sycl::global_ptr<float> J_gp(row.J);
          referenceNormal.store(0,J_gp);
          cross(projectedVertex, referenceNormal).store(1,J_gp);
        });
      });

      {
      const nd_range<1> ndr{size_of_group * number_of_groups, size_of_group};
      buffer<float,1> ob(reduceOutputBuffer, 32 * number_of_groups);
      q.submit([&](handler &cgh) {

        const auto out = ob.get_access<mode::write>(cgh);
        const auto J = ocl_trackingResult->get_access<mode::read>(cgh);
        using la_t = accessor<float, 1, mode::read_write, target::local>;
        la_t S{size_of_group * 32, cgh};

        cgh.parallel_for<kernels::reduceKernel>(ndr,[=](nd_item<1> ix) {
          uint blockIdx  = ix.get_group(0);
          uint blockDim  = ix.get_local_range(0);
          uint threadIdx = ix.get_local_id(0);

          uint gridDim   = ix.get_group_range(0);

          const uint sline = threadIdx;

          float         sums[32];
          float *jtj  = sums + 7;
          float *info = sums + 28;

          for (uint i = 0; i < 32; ++i)
            sums[i] = 0.0f;

          for (uint y = blockIdx; y < (uint)localimagesize.y(); y += gridDim) {
            for (uint x = sline; x < (uint)localimagesize.x(); x += blockDim) {
              const TrackData row = J[x + y * outputSize.x()];
              if (row.result < 1) {
                info[1] += row.result == -4 ? 1 : 0;
                info[2] += row.result == -5 ? 1 : 0;
                info[3] += row.result  > -4 ? 1 : 0;
                continue;
              }

              // Error part
              sums[0] += row.error * row.error;

              // JTe part
              for (int i = 0; i < 6; ++i)
                sums[i+1] += row.error * row.J[i];

              jtj[0]  += row.J[0] * row.J[0];
              jtj[1]  += row.J[0] * row.J[1];
              jtj[2]  += row.J[0] * row.J[2];
              jtj[3]  += row.J[0] * row.J[3];
              jtj[4]  += row.J[0] * row.J[4];
              jtj[5]  += row.J[0] * row.J[5];

              jtj[6]  += row.J[1] * row.J[1];
              jtj[7]  += row.J[1] * row.J[2];
              jtj[8]  += row.J[1] * row.J[3];
              jtj[9]  += row.J[1] * row.J[4];
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

          ix.barrier(cl::sycl::access::fence_space::local_space);

          // sum up columns and copy to global memory in the final 32 threads
          if (sline < 32) {
            for (unsigned i = 1; i < blockDim; ++i)
              S[sline] += S[i * 32 + sline];
            out[sline+blockIdx*32] = S[sline];
          }
        });
      });
      }
	   
      TooN::Matrix<TooN::Dynamic, TooN::Dynamic, float,
        TooN::Reference::RowMajor> values(reduceOutputBuffer,
          number_of_groups, 32);
	       
			for (int j = 1; j < number_of_groups; ++j) {
				values[0] += values[j];
			}

      if (updatePoseKernel(pose, reduceOutputBuffer, icp_threshold))
        break;
    }
  }
	
	return checkPoseKernel(pose, oldPose, reduceOutputBuffer, computationSize,
			track_threshold);
}

template <typename T>
inline float vs(const uint3 pos, const Volume<T> v) {
	return v.data.get()[(uint)pos.x() +
                      (uint)pos.y() * (uint)v.size.x() +
                      (uint)pos.z() * (uint)v.size.x() * (uint)v.size.y()].x();
}

template <typename T>
inline float interp(const float3 pos, const Volume<T> v) {
	const float3 scaled_pos
    {((float)pos.x() * (uint)v.size.x() / (float)v.dim.x()) - 0.5f,
     ((float)pos.y() * (uint)v.size.y() / (float)v.dim.y()) - 0.5f,
     ((float)pos.z() * (uint)v.size.z() / (float)v.dim.z()) - 0.5f};
//	float3 basef{0,0,0};
  float3 tmp = floor(scaled_pos);
	const int3 base{tmp.x(),tmp.y(),tmp.z()};
//	const float3 factor{cl::sycl::fract(scaled_pos, (float3 *) &basef)};
  const float3 factor = fmin(scaled_pos - floor(scaled_pos), 0x1.fffffep-1f);
  //float3 basef = cl::sycl::floor(scaled_pos);

  const int3 lower = max(base, int3{0,0,0});
  const int3 upper = min(base + int3{1,1,1},
                             int3{((uint)v.size.x())-1,((uint)v.size.y())-1,((uint)v.size.z())-1});
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
                  * factor.x()) * factor.y()) * factor.z()) * inv_32766;
}

template <typename T>
inline float3 grad(float3 pos, const Volume<T> v) {
	const float3 scaled_pos
    {((float)pos.x() * (uint)v.size.x() / (float)v.dim.x()) - 0.5f,
     ((float)pos.y() * (uint)v.size.y() / (float)v.dim.y()) - 0.5f,
     ((float)pos.z() * (uint)v.size.z() / (float)v.dim.z()) - 0.5f};
  float3 tmp = floor(scaled_pos);
	const int3 base{tmp.x(),tmp.y(),tmp.z()};
	//const float3 basef{0,0,0};
	//const float3 factor = (float3) fract(scaled_pos, (float3 *) &basef);
  // fract is absent; so use Khronos' definition:
  const float3 factor = fmin(scaled_pos - floor(scaled_pos), 0x1.fffffep-1f);
  //float3 basef = cl::sycl::floor(scaled_pos);

  const int3 vsm1((int)v.size.x()-1, (int)v.size.y()-1, (int)v.size.z()-1);
	const int3 lower_lower = max(base - int3{1,1,1}, int3{0,0,0});
	const int3 lower_upper = max(base,               int3{0,0,0});
	const int3 upper_lower = min(base + int3{1,1,1}, vsm1);
	const int3 upper_upper = min(base + int3{2,2,2}, vsm1);
	const int3 lower       = lower_upper;
	const int3 upper       = upper_lower;

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

	return gradient * float3{((float)v.dim.x()) / ((uint)v.size.x()),
							 ((float)v.dim.y()) / ((uint)v.size.y()),
							 ((float)v.dim.z()) / ((uint)v.size.z())} * (0.5f * inv_32766);
}

template <typename T>
float4 raycast(const Volume<T> v, const uint2 pos, const Matrix4 view,
               const float nearPlane, const float farPlane, const float step,
               const float largestep)
{
  const float3 origin = get_translation(view);
  const float3 direction = rotate(view, float3{pos.x(), pos.y(), 1.f});

	// intersect ray with a box
	//
	// www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
	// compute intersection of ray with all six bbox planes
  const float3 invR{1.0f/direction.x(), 1.0f/direction.y(), 1.0f/direction.z()};
  const float3 tbot = -1.0f * invR * origin;
	const float3 ttop = invR * (v.dim - origin);

  // re-order intersections to find smallest and largest on each axis
  const float3 tmin = cl::sycl::fmin(ttop, tbot);
  const float3 tmax = cl::sycl::fmax(ttop, tbot);

	// find the largest tmin and the smallest tmax
	const float largest_tmin =
    cl::sycl::fmax((float)cl::sycl::fmax((float)tmin.x(), (float)tmin.y()),
                   (float)cl::sycl::fmax((float)tmin.x(), (float)tmin.z()));
	const float smallest_tmax =
    cl::sycl::fmin((float)cl::sycl::fmin((float)tmax.x(), (float)tmax.y()),
                   (float)cl::sycl::fmin((float)tmax.x(), (float)tmax.z()));

	// check against near and far plane
	const float tnear = cl::sycl::fmax(largest_tmin, nearPlane);
	const float tfar  = cl::sycl::fmin(smallest_tmax, farPlane);

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

bool Kfusion::raycasting(float4 k, const float mu, const uint frame) {

  const float largestep = mu * 0.75f;

  if (frame > 2) {
    raycastPose = pose;
    const Matrix4 view = raycastPose * getInverseCameraMatrix(k);
    range<2> r{computationSize.x(), computationSize.y()};

    q.submit([&](handler &cgh) {
      using namespace cl::sycl::access;
      const auto pos3D  = ocl_vertex->get_access<mode::read_write>(cgh);
      const auto normal = ocl_normal->get_access<mode::read_write>(cgh);
      const auto v_data = ocl_volume_data->get_access<mode::read_write>(cgh);
      const auto v_size = this->volumeResolution;
      const auto v_dim  = this->volumeDimensions;
      const auto step   = this->step;
      cgh.parallel_for<kernels::raycastKernel>(r, [=](const item<2> ix) {
        Volume<cl::sycl::global_ptr<short2>> volume;
        volume.data = v_data; volume.size = v_size; volume.dim = v_dim;
        const uint2 pos{ix[0],ix[1]};
        const int sizex = ix.get_range()[0];

        const float4 hit =
          ::raycast(volume, pos, view, nearPlane, farPlane, step, largestep);
        const float3 test{hit.x(),hit.y(),hit.z()}; // as_float3(hit);
        const float3 invalid3{KFUSION_INVALID,KFUSION_INVALID,KFUSION_INVALID};

        if ((float)hit.w() > 0.0f) {
          pos3D[(uint)pos.x() + sizex * (uint)pos.y()] = test;
          const float3 surfNorm = grad(test,volume);
          if (length(surfNorm) == 0)
            normal[(int)pos.x() + sizex * pos.y()] = invalid3;
          else
            normal[(int)pos.x() + sizex * pos.y()] = normalize(surfNorm);
        } else {
          pos3D [(int)pos.x() + sizex * pos.y()] = float3{0,0,0};
          normal[(int)pos.x() + sizex * pos.y()] = invalid3;
        }
      });
    });
	}

	return false;
}

bool Kfusion::integration(float4 k, const uint integration_rate,
                          const float mu, const uint frame)
{
	bool doIntegrate = checkPoseKernel(pose, oldPose, reduceOutputBuffer,
			computationSize, track_threshold);

	if ((doIntegrate && (frame % integration_rate) == 0) || (frame <= 3)) {
		const Matrix4 invTrack = inverse(pose);
		const Matrix4 K = getCameraMatrix(k);
		const float3 delta = rotate(invTrack,
      float3{0, 0, (float)volumeDimensions.z() / (float)volumeResolution.z()});

    const range<2> r{volumeResolution.x(), volumeResolution.y()};
    q.submit([&](handler &cgh) {
      using namespace cl::sycl::access;
      const auto v_data = ocl_volume_data->get_access<mode::read_write>(cgh);
      const auto depth  = ocl_FloatDepth->get_access<mode::read>(cgh);
      const auto v_size = volumeResolution;
      const auto v_dim  = volumeDimensions;
      const auto cameraDelta = rotate(K,delta);
      const auto depthSize = computationSize;

      cgh.parallel_for<kernels::integrateKernel>(r, [=](const item<2> ix) {
        Volume<cl::sycl::global_ptr<short2>> vol;
        vol.data = v_data; // &v_data[0];
        vol.size = v_size;
        vol.dim  = v_dim;

        uint3 pix{ix[0],ix[1],0};
        const int sizex = ix.get_range()[0];

        float3 pos     = invTrack * posVolume(vol,pix);
        float3 cameraX = K * pos;

        for (pix.z() = 0; (uint)pix.z() < (uint)vol.size.z();
               pix.z() = (uint)pix.z()+1, pos += delta, cameraX += cameraDelta)
        {
          if ((float)pos.z() < 0.0001f) // some near plane constraint
            continue;

          const float2 pixel{cameraX.x()/cameraX.z() + 0.5f,
                             cameraX.y()/cameraX.z() + 0.5f};

          if ((float)pixel.x() < 0.0f ||
              (float)pixel.x() > (float)((uint)depthSize.x()-1) ||
              (float)pixel.y() < 0.0f ||
              (float)pixel.y() > (float)((uint)depthSize.y()-1))
            continue;

          const uint2 px{pixel.x(), pixel.y()};
          float depthpx = depth[(uint)px.x() + (uint)depthSize.x() * (uint)px.y()];

          if (depthpx == 0)
            continue;

          const float diff = (depthpx - cameraX.z()) *
               cl::sycl::sqrt(1+sq(pos.x()/pos.z()) + sq(pos.y()/pos.z()));

          if (diff > -mu)
          {
            const float sdf = cl::sycl::fmin(1.f, diff/mu);
            float2 data = getVolume(vol,pix);
            data.x() = clamp((float)(data.y()*data.x() + sdf)/((float)data.y() + 1),-1.f,1.f);
            data.y() = cl::sycl::fmin(((float)data.y())+1, maxweight);
            setVolume(vol,pix,data);
          }
        }
      });
    });

		doIntegrate = true;
	} else {
		doIntegrate = false;
	}

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
	if (!fDumpFile.is_open()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}

	const unsigned vol_size = volume.size.x() * volume.size.y() * volume.size.z();

	// Dump on file without the y component of the short2 variable
	for (unsigned int i = 0; i < vol_size; i++) {
		fDumpFile.write((char *) (volume.data + i), sizeof(short));
	}

	fDumpFile.close();
}

void Kfusion::renderVolume(uchar4 *out, const uint2 outputSize, const int frame,
                           const int raycast_rendering_rate, const float4 k,
                           const float largestep)
{
	if (frame % raycast_rendering_rate != 0) return;

  const range<2> r{computationSize.x(), computationSize.y()};
  const Matrix4 view = *this->viewPose * getInverseCameraMatrix(k);
  buffer<uchar4,1> ob(out,{(uint)outputSize.x() * (uint)outputSize.y()});

  q.submit([&](handler &cgh) {

    using namespace cl::sycl::access;
    const auto render = ob.get_access<mode::write>(cgh);
    const auto v_data = ocl_volume_data->get_access<mode::read_write>(cgh);
    const auto v_size = this->volumeResolution;
    const auto v_dim  = this->volumeDimensions;
    const auto step   = this->step;

    cgh.parallel_for<kernels::renderVolumeKernel>(r, [=](const item<2> ix)
    {
      Volume<cl::sycl::global_ptr<short2>> v;
      v.data = v_data; v.size = v_size; v.dim = v_dim;

      const uint2 pos{ix[0],ix[1]};
      const int sizex = ix.get_range()[0];
            auto   &r = render[ix[0] + sizex * ix[1]];

      const float4 hit=::raycast(v,pos,view,nearPlane,farPlane,step,largestep);

      if ((float)hit.w() > 0.0f) {
        const float3 test{hit.x(),hit.y(),hit.z()};
        const float3 surfNorm = grad(test,v);

        if (length(surfNorm) > 0) {
          const float3 diff = normalize(light - test);
          const float dir = cl::sycl::fmax(dot(normalize(surfNorm), diff), 0.f);
          const float3 col=clamp(float3(dir)+ambient,float3(0),float3(1)) * 255;
          r = uchar4{col.x(),col.y(),col.z(),0};
        } else {
          r = uchar4{0,0,0,0};
        }
      } else {
          r = uchar4{0,0,0,0};
      }
    });
  });
}

void Kfusion::renderTrack(uchar4 *out, const uint2 outputSize)
{
  const range<2> r{computationSize.x(), computationSize.y()};
  buffer<uchar4,1> ob(out,{(uint)outputSize.x() * (uint)outputSize.y()});

  q.submit([&](handler &cgh) {

    using namespace cl::sycl::access;
    const auto out  = ob.get_access<mode::write>(cgh);
    const auto data = ocl_trackingResult->get_access<mode::read>(cgh);

    cgh.parallel_for<kernels::renderTrackKernel>(r, [=](const item<2> ix)
    {
      const int   sizex = ix.get_range()[0];
            auto     &o =  out[ix[0] + sizex * ix[1]];
      const auto result = data[ix[0] + sizex * ix[1]].result;

      switch (result) {
        case  1: o = uchar4{128, 128, 128, 0}; break;
        case -1: o = uchar4{  0,   0,   0, 0}; break;
        case -2: o = uchar4{255,   0,   0, 0}; break;
        case -3: o = uchar4{  0, 255,   0, 0}; break;
        case -4: o = uchar4{  0,   0, 255, 0}; break;
        case -5: o = uchar4{255, 255,   0, 0}; break;
        default: o = uchar4{255, 128, 128, 0}; break;
      }
    });
  });
}

void Kfusion::renderDepth(uchar4 *out, const uint2 outputSize) {

  const range<2> r{computationSize.x(), computationSize.y()};
  buffer<uchar4,1> ob(out,{(uint)outputSize.x() * (uint)outputSize.y()});

  q.submit([&](handler &cgh) {

    using namespace cl::sycl::access;
    const auto out   = ob.get_access<mode::read_write>(cgh);
    const auto depth = ocl_FloatDepth->get_access<mode::read>(cgh);

    cgh.parallel_for<kernels::renderDepthKernel>(r, [=](const item<2> ix)
    {
      const int sizex = ix.get_range()[0];
            auto   &o =   out[ix[0] + sizex * ix[1]];
      const auto   &d = depth[ix[0] + sizex * ix[1]];

      if (d < nearPlane)
        o = uchar4{255,255,255,0};
      else {
        if (d > farPlane)
          o = uchar4{0,0,0,0};
        else {
          float h = (d - nearPlane) / (farPlane - nearPlane);
          h *= 6.0f;
          const int   sextant    = h;
          const float fract      = h - sextant;
          const float swift_half = 0.75f * 0.6667f; // 0.500025!! see vsf in gs2rgb
          const float mid1       = 0.25f + (swift_half*fract);
          const float mid2       = 0.75f - (swift_half*fract);
          // n.b. (char)(0.25*255) = 63  (and (char)(0.75*255) = 191)
          // This is to match the cpp version.
          // Same result as the simpler: (f*256)-1
          switch (sextant)
          {
            case 0: o = uchar4{191, 255*mid1, 63, 0}; break;
            case 1: o = uchar4{255*mid2, 191, 63, 0}; break;
            case 2: o = uchar4{63, 191, 255*mid1, 0}; break;
            case 3: o = uchar4{63, 255*mid2, 191, 0}; break;
            case 4: o = uchar4{255*mid1, 63, 191, 0}; break;
            case 5: o = uchar4{191, 63, 255*mid2, 0}; break;
          }
        }
      }
    });
  });
}

void synchroniseDevices() { q.wait(); }
