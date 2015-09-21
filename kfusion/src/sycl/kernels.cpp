/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */
#include <SYCL/sycl.hpp>
#include <dagr/dagr.hpp>
#include <kernels.h>

// input once
float * gaussian;

// inter-frame
Volume<short2 *> volume;
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

// sycl specific
cl::sycl::queue q(cl::sycl::intel_selector{});

buffer<float,1>      *ocl_gaussian             = NULL;

buffer<float3,1>     *ocl_vertex               = NULL;
buffer<float3,1>     *ocl_normal               = NULL;
buffer<short2,1>     *ocl_volume_data          = NULL;
buffer<uint16_t,1>   *ocl_depth_buffer         = NULL;

buffer<TrackData,1>  *ocl_trackingResult       = NULL;
buffer<float,1>      *ocl_FloatDepth           = NULL;
buffer<float,1>     **ocl_ScaledDepth          = NULL;
buffer<float3,1>    **ocl_inputVertex          = NULL;
buffer<float3,1>    **ocl_inputNormal          = NULL;
float *reduceOutputBuffer = NULL;

// reduction parameters
static const size_t size_of_group    = 64;
static const size_t number_of_groups = 8;

uint2 computationSizeBkp = make_uint2(0, 0);
uint2 outputImageSizeBkp = make_uint2(0, 0);

void Kfusion::languageSpecificConstructor() {

  const auto csize = computationSize.x() * computationSize.y();

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

	reduceOutputBuffer = (float*) malloc(number_of_groups * 32 * sizeof(float));

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

  ocl_gaussian = new buffer<float,1>(gaussian, range<1>{gaussianS});
	// ********* END : Generate the gaussian *************

  const auto vsize = volumeResolution.x() *
                     volumeResolution.y() * volumeResolution.z();
  ocl_volume_data = new buffer<short2>(range<1>{vsize});

	volume.init(volumeResolution, volumeDimensions);
	reset();
}

Kfusion::~Kfusion() {

	if (reduceOutputBuffer)
		free(reduceOutputBuffer);
	reduceOutputBuffer = NULL;

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
	if (ocl_depth_buffer) {
		delete ocl_depth_buffer;
		ocl_depth_buffer = NULL;
  }
	if (ocl_trackingResult) {
		delete ocl_trackingResult;
		ocl_trackingResult = NULL;
  }

//

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
  uint3 &vr = volumeResolution; // declared in kernels.h
  const auto r = range<3>{vr.x(),vr.y(),vr.x()};
	dagr::run<initVolumeKernel,0>(q,r,*ocl_volume_data);
}

void init()  { } // stub
void clean() { } // stub

struct initVolumeKernel {

template <typename T>
static void k(item<3> ix, T *data)
{
  uint x = ix[0]; uint y = ix[1]; uint z = ix[2];
  uint3 size{ix.get_range()[0], ix.get_range()[1], ix.get_range()[2]};
  float2 d{1.0f,0.0f};

  data[x + y * size.x() + z * size.x() * size.y()] =
    short2{d.x() * 32766.0f, d.y()};
}

}; // struct

struct bilateralFilterKernel {

template <typename T>
static void k(item<2> ix, T *out, const T *in, const T *gaussian,
              const float e_d, const int r)
{
  /*const*/ uint2 pos{ix[0],ix[1]};
  /*const*/ uint2 size{ix.get_range()[0], ix.get_range()[1]};

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
}

};

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
	return float2{d.x() * 0.00003051944088f, d.y()}; //  / 32766.0f
}

struct depth2vertexKernel {

// vertex is actually an array of float3 (T == float3).
template <typename T, typename U>
static void k(item<2> ix, T *vertex, const U *depth, const Matrix4 invK)
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

struct vertex2normalKernel {

// normal and vertex are actually arrays of float3
template <typename T>
static void k(item<2> ix, T *normal, const T *verte_)
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
    const float3 invalid3{INVALID,INVALID,INVALID};
    normal[pixel.x() + ix.get_range()[0] * pixel.y()] = invalid3;
    return;
  }

  const float3 dxv = right - left;
  const float3 dyv = down  - up;
  normal[pixel.x() + ix.get_range()[0] * pixel.y()] = normalize(cross(dyv,dxv));
}

}; // struct

struct reduceKernel {

template <typename T, typename U, typename V>
static void k(nd_item<1> ix, T *out, const U *J_,
              /*const*/ uint2 JSize, /*const*/ uint2 size, V *S)
{
  const TrackData *J = J_;  // See const_vec_ptr.cpp

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

  for (uint y = blockIdx; y < size.y(); y += gridDim) {
    for (uint x = sline; x < size.x(); x += blockDim) {
      const TrackData row = J[x + y * JSize.x()];
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
}

}; // struct

struct trackKernel {
// inVertex, inNormal, refVertex, and refNormal are really float3 arrays.
template <typename T, typename U>
static void k(item<2> ix, T *output,      /*const*/ uint2 outputSize,
              const U *inVerte_,          const U *inNorma_,
              const U *refVertex,         const U *refNorma_,
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
    return;
  }
  if (dot(projectedNormal,referenceNormal) < normal_threshold) {
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

struct mm2metersKernel {

template <typename T, typename U>
static void k(item<2> ix, T *depth, /*const*/ uint2 depthSize,
              const U *in, /*const*/ uint2 inSize, const int ratio)
{
  uint2 pixel{ix[0],ix[1]};
  depth[pixel.x() + depthSize.x() * pixel.y()] =
    in[pixel.x() * ratio + inSize.x() * pixel.y() * ratio] / 1000.0f;
//  depth[ix] = in[ ix.get()*ratio ] / 1000.0f;
}

}; // struct

struct halfSampleRobustImageKernel {

template <typename T>
static void k(item<2> ix, T *out, const T *in,
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

struct integrateKernel {

template <typename I, typename T, typename U>
static void k(I ix, T *v_data, const uint3 v_size, const float3 v_dim,
              const U *depth, /*const*/ uint2 depthSize,
              const Matrix4 invTrack, const Matrix4 K, const float mu,
              const float maxweight, const float3 delta,
              const float3 cameraDelta)
{
  Volume<T *> vol;
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

struct raycastKernel {

// Although T is instantiated float, pos3D and normal are targeting float3 data
// Actually, T is instantiated as float3. This is due to the declaration of
// ocl_vertex, which was given a float3 type, rather than the uptyped OpenCL buf
template <typename T, typename U>
static void k(item<2> ix, T *pos3D, T *normal, U *v_data,
              const uint3 v_size, const float3 v_dim, const Matrix4 view,
              const float nearPlane, const float farPlane,
              const float step, const float largestep)
{
  /*const*/ Volume<U *> volume;//{v_size,v_dim,v_data};
  volume.data = &v_data[0]; volume.size = v_size; volume.dim = v_dim;
  uint2 pos{ix[0],ix[1]};
  const int sizex = ix.get_range()[0];

  /*const*/ float4 hit =
    raycast(volume, pos, view, nearPlane, farPlane, step, largestep);
  const float3 test{hit.x(),hit.y(),hit.z()}; // as_float3(hit);

  // The C++ version just sets the normal's x value to INVALID. This is a
  // better approach - also used by the OpenCL version.
  const float3 invalid3{INVALID,INVALID,INVALID};

  if (hit.w() > 0.0f) {
    pos3D[pos.x() + sizex * pos.y()] = test;
    float3 surfNorm = grad(test,volume);
    if (cl::sycl::length(surfNorm) == 0)
      normal[pos.x() + sizex * pos.y()] = invalid3;
    else
      normal[pos.x() + sizex * pos.y()] = cl::sycl::normalize(surfNorm);
  }
  else {
    pos3D [pos.x() + sizex * pos.y()] = float3{0,0,0};
    normal[pos.x() + sizex * pos.y()] = invalid3;
  }

}

}; // struct

bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold)
{
	bool res = false;
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

	return res;
}

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output,
		uint2 imageSize, float track_threshold) {

	// Check the tracking result, and go back to the previous camera position if necessary

	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

	if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2)
			|| (values(0, 28) / (imageSize.x() * imageSize.y()) < track_threshold)) {
		pose = oldPose;
		return false;
	} else {
		return true;
	}

}

struct renderDepthKernel {

template <typename T, typename U>   // templates abstract over address spaces
static void k(item<2> ix, T *out, U const *depth,
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
      const int   sextant    = (int)h;
      const float fract      = h - sextant;
      const float swift_half = 0.75f * 0.6667f; // 0.500025!! see vsf in gs2rgb
      const float mid1       = 0.25f + (swift_half*fract);
      const float mid2       = 0.75f - (swift_half*fract);
// n.b. (char)(0.25*255) = 63  (and (char)(0.75*255) = 191) This is to match
// the cpp version. Same result as the simpler: (f*256)-1
			switch (sextant)
			{
        case 0: out[posx + sizex * posy] = uchar4{191, 255*mid1, 63, 0}; break;
        case 1: out[posx + sizex * posy] = uchar4{255*mid2, 191, 63, 0}; break;
        case 2: out[posx + sizex * posy] = uchar4{63, 191, 255*mid1, 0}; break;
        case 3: out[posx + sizex * posy] = uchar4{63, 255*mid2, 191, 0}; break;
        case 4: out[posx + sizex * posy] = uchar4{255*mid1, 63, 191, 0}; break;
        case 5: out[posx + sizex * posy] = uchar4{191, 63, 255*mid2, 0}; break;
			}
		}
	}
}

}; // struct renderDepthKernel

struct renderTrackKernel {

template <typename T, typename U>
static void k(item<2> ix, T * out, const U * data)
{
	const int posx  = ix[0];
	const int posy  = ix[1];
  const int sizex = ix.get_range()[0];

	switch (data[posx + sizex * posy].result) {
		case  1: out[posx + sizex * posy] = uchar4{128, 128, 128, 0}; break;
		case -1: out[posx + sizex * posy] = uchar4{  0,   0,   0, 0}; break;
		case -2: out[posx + sizex * posy] = uchar4{255,   0,   0, 0}; break;
		case -3: out[posx + sizex * posy] = uchar4{  0, 255,   0, 0}; break;
		case -4: out[posx + sizex * posy] = uchar4{  0,   0, 255, 0}; break;
		case -5: out[posx + sizex * posy] = uchar4{255, 255,   0, 0}; break;
		default: out[posx + sizex * posy] = uchar4{255, 128, 128, 0}; break;
	}
}

}; // struct

struct renderVolumeKernel {

template <typename T, typename U>
static void k(item<2> ix, T *render, U *v_data, const uint3 v_size,
              const float3 v_dim, const Matrix4 view,
              const float nearPlane, const float farPlane,
              const float step, const float largestep,
              const float3 light, const float3 ambient)
{
	/*const*/ Volume<U *> v;//{v_size,v_dim,v_data};
  v.data = v_data; v.size = v_size; v.dim = v_dim;

	/*const*/ uint2 pos{ix[0],ix[1]};
  const int sizex = ix.get_range()[0];

  float4 hit = raycast(v, pos, view, nearPlane, farPlane, step, largestep);

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

bool Kfusion::preprocessing(const uint16_t *inputDepth, /*const*/ uint2 inSize)
{
	// bilateral_filter(ScaledDepth[0], inputDepth, inputSize , gaussian, e_delta, radius);
	uint2 outSize = computationSize;
	const uint inSize_x  = inSize.x(); const uint inSize_y = inSize.y();
	const uint outSize_x = computationSize.x();
  const uint outSize_y = computationSize.y();

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

	if (computationSizeBkp.x() < inSize.x() ||
      computationSizeBkp.y() < inSize.y() || ocl_depth_buffer == NULL) {
		computationSizeBkp = make_uint2(inSize.x(), inSize.y());
		if (ocl_depth_buffer != NULL) {
      delete ocl_depth_buffer;
      ocl_depth_buffer = NULL;
		}
    auto in_sz = range<1>{inSize.x() * inSize.y()};
		ocl_depth_buffer = new buffer<uint16_t,1>(inputDepth, in_sz);
	}

    auto r = range<2>{outSize.x(),outSize.y()};

    dagr::run<mm2metersKernel,0>(q, r, *ocl_FloatDepth, outSize,
                                 dagr::ro(*ocl_depth_buffer), inSize, ratio);
    delete ocl_depth_buffer; ocl_depth_buffer = NULL; // debug only

    dagr::run<bilateralFilterKernel,0>(q, r, *ocl_ScaledDepth[0],
                                       dagr::ro(*ocl_FloatDepth),
                                       dagr::ro(*ocl_gaussian),e_delta,radius);
	return true;
}

bool Kfusion::tracking(float4 k, float icp_threshold, uint tracking_rate,
		uint frame) {

	if (frame % tracking_rate != 0)
		return false;

	// half sample the input depth maps into the pyramid levels
	for (unsigned int i = 1; i < iterations.size(); ++i) {
		cl::sycl::uint2 outSize{computationSize.x() / (int) ::pow(2, i),
                            computationSize.y() / (int) ::pow(2, i)};

    auto r   = range<2>{outSize.x(),outSize.y()};
    auto out = dagr::wo(*ocl_ScaledDepth[i  ]);
    auto in  = dagr::ro(*ocl_ScaledDepth[i-1]);
		uint2 inSize{outSize.x()*2,outSize.y()*2}; // Seems redundant
    dagr::run<halfSampleRobustImageKernel,0>(q,r,out,in,inSize,e_delta*3,1);
	}

	// prepare the 3D information from the input depth maps
	uint2 localimagesize = computationSize;
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		float4 tmp{k / float(1 << i)};
		Matrix4 invK = getInverseCameraMatrix(tmp);   // Needs a non-const (tmp)
    const range<2>  imageSize{localimagesize.x(),localimagesize.y()};
    dagr::run<depth2vertexKernel,0>(q, imageSize,
               *ocl_inputVertex[i], dagr::ro(*ocl_ScaledDepth[i]), invK);

    const range<2>  imageSize2{localimagesize.x(),localimagesize.y()};
    dagr::run<vertex2normalKernel,0>(q, imageSize2,
               *ocl_inputNormal[i], dagr::wo(*ocl_inputVertex[i]));

		localimagesize = make_uint2(localimagesize.x() / 2, localimagesize.y() / 2);
	}

	oldPose = pose;
	const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

  // iterations: a vector<int> set to {10,5,4} in Kfusion ctor (kernels.h)
  for (int level = iterations.size() - 1; level >= 0; --level) {
    const int csize_x = computationSize.x();
    const int csize_y = computationSize.y();
    const int pow2l = pow(2, level);
		uint2 localimagesize = make_uint2(csize_x / pow2l, csize_y / pow2l);

    for (int i = 0; i < iterations[level]; ++i) {  // i<4,i<5,i<10
      range<2> imageSize{localimagesize.x(),localimagesize.y()};
      dagr::run<trackKernel,0>(q,imageSize,*ocl_trackingResult,computationSize,
        dagr::ro(*ocl_inputVertex[level]), dagr::ro(*ocl_inputNormal[level]),
        dagr::ro(*ocl_vertex),             dagr::ro(*ocl_normal),
        pose,projectReference,dist_threshold,normal_threshold);

      const    range<1> nitems{size_of_group * number_of_groups};
      const nd_range<1> ndr{nd_range<1>(nitems, range<1>{size_of_group})};
      dagr::run<reduceKernel,0>(q,ndr,
        dagr::wo(buffer<float,1>(reduceOutputBuffer,
                                 range<1>{32 * number_of_groups})),
        dagr::ro(*ocl_trackingResult), computationSize, localimagesize,
        dagr::lo<float>(size_of_group * 32));

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
  /*const*/ float3 factor = // fract is absent; so use Khronos' definition:
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
float4 raycast(/*const*/ Volume<T> v, /*const*/ uint2 pos, const Matrix4 view,
               const float nearPlane, const float farPlane, const float step,
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

bool Kfusion::raycasting(float4 k, float mu, uint frame) {

	bool doRaycast = false;
	float largestep = mu * 0.75f;

	if (frame > 2) {
		raycastPose = pose;
		const Matrix4 view = raycastPose * getInverseCameraMatrix(k);
    range<2> RaycastglobalWorksize{computationSize.x(), computationSize.y()};

    dagr::run<raycastKernel,0>(q,RaycastglobalWorksize,
      *ocl_vertex,*ocl_normal,*ocl_volume_data,
      volumeResolution,volumeDimensions,view,nearPlane,farPlane,step,largestep);
	}

	return doRaycast;
}

bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame)
{
	bool doIntegrate = checkPoseKernel(pose, oldPose, reduceOutputBuffer,
			computationSize, track_threshold);

	if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
    range<2> globalWorksize{volumeResolution.x(), volumeResolution.y()};
		const Matrix4 invTrack = inverse(pose);
		const Matrix4 K = getCameraMatrix(k);
		const float3 delta = myrotate(invTrack,
				float3{0, 0, volumeDimensions.z() / volumeResolution.z()});

    // The SYCL lambda for integrateKernel demonstrates pre-DAGR verbosity
    dagr::run<integrateKernel,0>(q, globalWorksize,
      *ocl_volume_data, volumeResolution, volumeDimensions,
      dagr::ro(*ocl_FloatDepth),
      computationSize, invTrack, K, mu, maxweight, delta, myrotate(K, delta));

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
	if (fDumpFile == NULL) {
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

void Kfusion::renderVolume(uchar4 * out, uint2 outputSize, int frame,
                           int raycast_rendering_rate, float4 k,
                           float largestep)
{
	if (frame % raycast_rendering_rate != 0) return;

  range<2> globalWorksize{computationSize.x(), computationSize.y()};
  Matrix4 view = *(this->viewPose) * getInverseCameraMatrix(k);
  dagr::run<renderVolumeKernel,0>(q,globalWorksize,
    dagr::wo(buffer<uchar4,1>(out,range<1>{outputSize.x() * outputSize.y()})),
    *ocl_volume_data,volumeResolution,volumeDimensions,view,nearPlane,farPlane,
    step,largestep,light,ambient);
}

void Kfusion::renderTrack(uchar4 * out, uint2 outputSize)
{
  range<2> globalWorksize{computationSize.x(), computationSize.y()};
  dagr::run<renderTrackKernel,0>(q,globalWorksize,
    dagr::wo(buffer<uchar4,1>(out,range<1>{outputSize.x() * outputSize.y()})),
    dagr::ro(*ocl_trackingResult));
}

void Kfusion::renderDepth(uchar4 * out, uint2 outputSize) {

  range<2> globalWorksize{computationSize.x(), computationSize.y()};
  dagr::run<renderDepthKernel,0>(q,globalWorksize,
    dagr::wo(buffer<uchar4,1>(out,range<1>{outputSize.x() * outputSize.y()})),
    dagr::ro(*ocl_FloatDepth), nearPlane, farPlane);
}

void synchroniseDevices() { q.wait(); }
