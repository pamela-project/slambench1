#ifndef _SYCL_VEC_MATHS_
#define _SYCL_VEC_MATHS_

/*

  Copyright (c) 2015 Paul Keir, University of the West of Scotland
  This code is licensed under the MIT License.

*/

#include <SYCL/sycl.hpp>
using cl::sycl::accessor; using cl::sycl::buffer;  using cl::sycl::handler;
using cl::sycl::nd_range; using cl::sycl::range;
using cl::sycl::nd_item;  using cl::sycl::item;
namespace sycl_a = cl::sycl::access;

using cl::sycl::float2;    using cl::sycl::float3; using cl::sycl::float4;
using cl::sycl::int2;      using cl::sycl::int3;
using cl::sycl::uint2;     using cl::sycl::uint3;
using cl::sycl::short2;

using cl::sycl::uchar3;    using cl::sycl::uchar4;
using cl::sycl::clamp;     using cl::sycl::min;    using cl::sycl::max;
using cl::sycl::normalize; using cl::sycl::cross;  using cl::sycl::length;
using cl::sycl::dot;

inline float2 make_float2(float x, float y         )  { return float2{x,y}; }
inline float3 make_float3(float x, float y, float z)  { return float3{x,y,z}; }
inline float3 make_float3(float4 a) { return float3{a.x(), a.y(), a.z()}; }
inline float4 make_float4(float x, float y, float z, float w)  {
  return float4{x,y,z,w};
}
inline
uint3  make_uint3(unsigned x, unsigned y, unsigned z) { return uint3{x,y,z}; }
inline uchar3 make_uchar3(unsigned char a, unsigned char b, unsigned char c) {
  return uchar3{a,b,c};
}
inline uchar4 make_uchar4(unsigned char a, unsigned char b,
                          unsigned char c, unsigned char d) {
  return uchar4{a,b,c,d};
}
inline int2   make_int2(int x, int y)                 { return int2{x,y};    }
inline int3   make_int3(int x, int y, int z)          { return int3{x,y,z};  }
inline int3   make_int3(float3 f)  { return int3{f.x(),f.y(),f.z()}; }
inline int3   make_int3(uint3 s)   { return int3{s.x(),s.y(),s.z()}; }
inline short2 make_short2(short x, short y)           { return short2{x,y}; }
inline float2 make_float2(float f) { return make_float2(f,f); }
inline float3 make_float3(float f) { return make_float3(f,f,f); }
inline float4 make_float4(float s) { return make_float4(s,s,s,s); }
inline float4 make_float4(float3 a, float w) {
  return float4{a.x(), a.y(), a.z(), w};
}

inline uint2  make_uint2(uint x, uint y)   { return uint2{x,y}; }
inline uint3  make_uint3(uint s)   { return make_uint3(s,s,s); }
inline int2   make_int2(int s)     { return int2{s,s}; }
inline int3   make_int3(int s)     { return make_int3(s,s,s); }
inline uint2  make_uint2(uint s)   { return uint2{s,s}; }
inline uint2  make_uint2(int2 a)   { return uint2{uint(a.x()), uint(a.y())}; }

inline float3 floorf(float3 v) {
  return float3{floorf(v.x()),floorf(v.y()),floorf(v.z())};
}
inline float  fracf(float v)  { return v - floorf(v); }
inline float3 fracf(float3 v) {
	return float3(fracf(v.x()), fracf(v.y()), fracf(v.z()));
}
inline float3 fminf(float3 a, float3 b) {
	return float3{fminf(a.x(), b.x()), fminf(a.y(), b.y()), fminf(a.z(), b.z())};
}
inline float3 fmaxf(float3 a, float3 b) {
	return float3{fmaxf(a.x(), b.x()), fmaxf(a.y(), b.y()), fmaxf(a.z(), b.z())};
}
inline float  min(float3 a) { return fminf(a.x(), fminf(a.y(), a.z())); }
inline uint   max(uint3 a)  { 	
	return max(a.get_value(0), max(a.get_value(1), a.get_value(2))); 
}
inline float3 operator*(float b, float3 a) {
	return float3{b*a.x(), b*a.y(), b*a.z()};
}
inline uint2 operator*(uint b, uint2 a) { return uint2{b * a.x(), b * a.y()}; }
// SYCL's dot, length, normalize etc. don't work on host 
inline float my_dot(float3 a, float3 b) {
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
inline float my_dot(float4 a, float4 b) {
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}

#endif // _SYCL_VEC_MATHS_
