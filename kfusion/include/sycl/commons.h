/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef _SYCL_COMMONS_
#define _SYCL_COMMONS_

#if defined(__GNUC__)
// circumvent packaging problems in gcc 4.7.0
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

// need c headers for __int128 and uint16_t
#include <limits.h>
#endif
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <iterator>

// Internal dependencies
#include <default_parameters.h> // (CUDA) includes vector_types.h + cutil_math.h









//External dependencies
#undef isnan
#undef isfinite
#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/GR_SVD.h>

////////////////////////// MATh STUFF //////////////////////

#define INVALID -2
// DATA TYPE

inline
bool is_file(std::string path) {
	struct stat buf;
	stat(path.c_str(), &buf);
	return S_ISREG(buf.st_mode);
}

template<typename T>
std::string NumberToString(T Number, int width = 6) {
	std::ostringstream ss;
	ss << std::setfill('0') << std::setw(width) << Number;
	return ss.str();
}

template<typename T>
void read_input(std::string inputfile, T * in) {
	size_t isize;
	std::ifstream file(inputfile.c_str(),
			std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open()) {
		isize = file.tellg();
		file.seekg(0, std::ios::beg);
		file.read((char*) in, isize);
		file.close();
	} else {
		std::cout << "File opening failed : " << inputfile << std::endl;
		exit(1);
	}
}

inline float sq(float r) {
	return r * r;
}

inline uchar4 gs2rgb(double h) {
	uchar4 rgb;
	double v;
	double r, g, b;
	v = 0.75;
	if (v > 0) {
		double m;
		double sv;
		int sextant;
		double fract, vsf, mid1, mid2;
		m = 0.25;
		sv = 0.6667;
		h *= 6.0;
		sextant = (int) h;
		fract = h - sextant;
		vsf = v * sv * fract;
		mid1 = m + vsf;
		mid2 = v - vsf;
		switch (sextant) {
		case 0:
			r = v;
			g = mid1;
			b = m;
			break;
		case 1:
			r = mid2;
			g = v;
			b = m;
			break;
		case 2:
			r = m;
			g = v;
			b = mid1;
			break;
		case 3:
			r = m;
			g = mid2;
			b = v;
			break;
		case 4:
			r = mid1;
			g = m;
			b = v;
			break;
		case 5:
			r = v;
			g = m;
			b = mid2;
			break;
		default:
			r = 0;
			g = 0;
			b = 0;
			break;
		}
	}
	rgb.x() = r * 255;
	rgb.y() = g * 255;
	rgb.z() = b * 255;
	rgb.w() = 0; // Only for padding purposes 
	return rgb;
}

template <typename T>
struct Volume {
	uint3 size;
	float3 dim;
  T data;
  //  short2 * data;

	Volume() {
		size = make_uint3(0);
		dim = make_float3(1);
		data = NULL;
	}

	float2 operator[](/*const*/ uint3 & pos) /*const*/ {
		/*const*/ short2 d = data[pos.x() + pos.y() * size.x() + pos.z() * size.x() * size.y()];
		return make_float2(d.x() * 0.00003051944088f, d.y()); //  / 32766.0f
	}

	float v(/*const*/ uint3 & pos) /*const*/ {
		return operator[](pos).x();
	}

	float vs(/*const*/ uint3 & pos) /*const*/ {
		return data[pos.x() + pos.y() * size.x() + pos.z() * size.x() * size.y()].x();
	}
	inline float vs2(const uint x, const uint y, const uint z) /*const*/ {
		return data[x + y * size.x() + z * size.x() * size.y()].x();
	}

	void setints(const unsigned x, const unsigned y, const unsigned z,
               /*const*/ float2 &d) {
    data[x + y * size.x() + z * size.x() * size.y()] =
      make_short2(d.x() * 32766.0f, d.y());
	}

	void set(/*const*/ uint3 & pos, /*const*/ float2 & d) {
		data[pos.x() + pos.y() * size.x() + pos.z() * size.x() * size.y()] =
      make_short2(d.x() * 32766.0f, d.y());
	}
	float3 pos(/*const*/ uint3 & p) /*const*/ {
		return make_float3((p.x() + 0.5f) * dim.x() / size.x(),
				(p.y() + 0.5f) * dim.y() / size.y(), (p.z() + 0.5f) * dim.z() / size.z());
	}

	float interp(/*const*/ float3 & pos) /*const*/ {

		const float3 scaled_pos = make_float3((pos.x() * size.x() / dim.x()) - 0.5f,
				(pos.y() * size.y() / dim.y()) - 0.5f,
				(pos.z() * size.z() / dim.z()) - 0.5f);
		const int3 base = make_int3(floorf(scaled_pos));
		/*const*/ float3 factor = fracf(scaled_pos);
		/*const*/ int3 lower = max(base, make_int3(0));
		/*const*/ int3 upper = min(base + make_int3(1),
				make_int3(size) - make_int3(1));
		return (((vs2(lower.x(), lower.y(), lower.z()) * (1 - factor.x())
				+ vs2(upper.x(), lower.y(), lower.z()) * factor.x()) * (1 - factor.y())
				+ (vs2(lower.x(), upper.y(), lower.z()) * (1 - factor.x())
						+ vs2(upper.x(), upper.y(), lower.z()) * factor.x()) * factor.y())
				* (1 - factor.z())
				+ ((vs2(lower.x(), lower.y(), upper.z()) * (1 - factor.x())
						+ vs2(upper.x(), lower.y(), upper.z()) * factor.x())
						* (1 - factor.y())
						+ (vs2(lower.x(), upper.y(), upper.z()) * (1 - factor.x())
								+ vs2(upper.x(), upper.y(), upper.z()) * factor.x())
								* factor.y()) * factor.z()) * 0.00003051944088f;

	}

	float3 grad(/*const*/ float3 & pos) /*const*/ {
		const float3 scaled_pos = make_float3((pos.x() * size.x() / dim.x()) - 0.5f,
				(pos.y() * size.y() / dim.y()) - 0.5f,
				(pos.z() * size.z() / dim.z()) - 0.5f);
		const int3 base = make_int3(floorf(scaled_pos));
		/*const*/ float3 factor = fracf(scaled_pos);
		/*const*/ int3 lower_lower = max(base - make_int3(1), make_int3(0));
		/*const*/ int3 lower_upper = max(base, make_int3(0));
		/*const*/ int3 upper_lower = min(base + make_int3(1),
				make_int3(size) - make_int3(1));
		/*const*/ int3 upper_upper = min(base + make_int3(2),
				make_int3(size) - make_int3(1));
		/*const*/ int3 & lower = lower_upper;
		/*const*/ int3 & upper = upper_lower;

		float3 gradient;

		gradient.x() = (((vs2(upper_lower.x(), lower.y(), lower.z())
				- vs2(lower_lower.x(), lower.y(), lower.z())) * (1 - factor.x())
				+ (vs2(upper_upper.x(), lower.y(), lower.z())
						- vs2(lower_upper.x(), lower.y(), lower.z())) * factor.x())
				* (1 - factor.y())
				+ ((vs2(upper_lower.x(), upper.y(), lower.z())
						- vs2(lower_lower.x(), upper.y(), lower.z())) * (1 - factor.x())
						+ (vs2(upper_upper.x(), upper.y(), lower.z())
								- vs2(lower_upper.x(), upper.y(), lower.z()))
								* factor.x()) * factor.y()) * (1 - factor.z())
				+ (((vs2(upper_lower.x(), lower.y(), upper.z())
						- vs2(lower_lower.x(), lower.y(), upper.z())) * (1 - factor.x())
						+ (vs2(upper_upper.x(), lower.y(), upper.z())
								- vs2(lower_upper.x(), lower.y(), upper.z()))
								* factor.x()) * (1 - factor.y())
						+ ((vs2(upper_lower.x(), upper.y(), upper.z())
								- vs2(lower_lower.x(), upper.y(), upper.z()))
								* (1 - factor.x())
								+ (vs2(upper_upper.x(), upper.y(), upper.z())
										- vs2(lower_upper.x(), upper.y(), upper.z()))
										* factor.x()) * factor.y()) * factor.z();

		gradient.y() = (((vs2(lower.x(), upper_lower.y(), lower.z())
				- vs2(lower.x(), lower_lower.y(), lower.z())) * (1 - factor.x())
				+ (vs2(upper.x(), upper_lower.y(), lower.z())
						- vs2(upper.x(), lower_lower.y(), lower.z())) * factor.x())
				* (1 - factor.y())
				+ ((vs2(lower.x(), upper_upper.y(), lower.z())
						- vs2(lower.x(), lower_upper.y(), lower.z())) * (1 - factor.x())
						+ (vs2(upper.x(), upper_upper.y(), lower.z())
								- vs2(upper.x(), lower_upper.y(), lower.z()))
								* factor.x()) * factor.y()) * (1 - factor.z())
				+ (((vs2(lower.x(), upper_lower.y(), upper.z())
						- vs2(lower.x(), lower_lower.y(), upper.z())) * (1 - factor.x())
						+ (vs2(upper.x(), upper_lower.y(), upper.z())
								- vs2(upper.x(), lower_lower.y(), upper.z()))
								* factor.x()) * (1 - factor.y())
						+ ((vs2(lower.x(), upper_upper.y(), upper.z())
								- vs2(lower.x(), lower_upper.y(), upper.z()))
								* (1 - factor.x())
								+ (vs2(upper.x(), upper_upper.y(), upper.z())
										- vs2(upper.x(), lower_upper.y(), upper.z()))
										* factor.x()) * factor.y()) * factor.z();

		gradient.z() = (((vs2(lower.x(), lower.y(), upper_lower.z())
				- vs2(lower.x(), lower.y(), lower_lower.z())) * (1 - factor.x())
				+ (vs2(upper.x(), lower.y(), upper_lower.z())
						- vs2(upper.x(), lower.y(), lower_lower.z())) * factor.x())
				* (1 - factor.y())
				+ ((vs2(lower.x(), upper.y(), upper_lower.z())
						- vs2(lower.x(), upper.y(), lower_lower.z())) * (1 - factor.x())
						+ (vs2(upper.x(), upper.y(), upper_lower.z())
								- vs2(upper.x(), upper.y(), lower_lower.z()))
								* factor.x()) * factor.y()) * (1 - factor.z())
				+ (((vs2(lower.x(), lower.y(), upper_upper.z())
						- vs2(lower.x(), lower.y(), lower_upper.z())) * (1 - factor.x())
						+ (vs2(upper.x(), lower.y(), upper_upper.z())
								- vs2(upper.x(), lower.y(), lower_upper.z()))
								* factor.x()) * (1 - factor.y())
						+ ((vs2(lower.x(), upper.y(), upper_upper.z())
								- vs2(lower.x(), upper.y(), lower_upper.z()))
								* (1 - factor.x())
								+ (vs2(upper.x(), upper.y(), upper_upper.z())
										- vs2(upper.x(), upper.y(), lower_upper.z()))
										* factor.x()) * factor.y()) * factor.z();

		return gradient
				* make_float3(dim.x() / size.x(), dim.y() / size.y(), dim.z() / size.z())
				* (0.5f * 0.00003051944088f);
	}

	void init(uint3 s, float3 d) {
		size = s;
		dim = d;
		data = (short2 *) malloc(size.x() * size.y() * size.z() * sizeof(short2));
		assert(data != NULL);

	}

	void release() {
		free(data);
		data = NULL;
	}
};

typedef struct sMatrix4 {
	float4 data[4];
} Matrix4;

inline float3 get_translation(/*const*/ Matrix4 view) {
	return make_float3(view.data[0].w(), view.data[1].w(), view.data[2].w());
}

struct TrackData {
	int result;
	float error;
	float J[6];
};

// SYCL's host dot implementation is missing
inline float3 operator*(/*const*/ Matrix4 & M, const float3 & v) {
	return make_float3(my_dot(make_float3(M.data[0]), v) + M.data[0].w(),
                     my_dot(make_float3(M.data[1]), v) + M.data[1].w(),
                     my_dot(make_float3(M.data[2]), v) + M.data[2].w());
}

inline float3 rotate(const Matrix4 & M, const float3 & v) {
  return make_float3(my_dot(make_float3(M.data[0]), v),
                     my_dot(make_float3(M.data[1]), v),
                     my_dot(make_float3(M.data[2]), v));
}

inline Matrix4 getCameraMatrix(/*const*/ float4 & k) {
	Matrix4 K;
	K.data[0] = make_float4(k.x(), 0,     k.z(), 0);
	K.data[1] = make_float4(0,     k.y(), k.w(), 0);
	K.data[2] = make_float4(0,     0,     1,     0);
	K.data[3] = make_float4(0,     0,     0,     1);
	return K;
}

inline Matrix4 getInverseCameraMatrix(/*const*/ float4 & k) {
	Matrix4 invK;
	invK.data[0] = make_float4(1.0f / k.x(), 0,            -k.z() / k.x(), 0);
	invK.data[1] = make_float4(0,            1.0f / k.y(), -k.w() / k.y(), 0);
	invK.data[2] = make_float4(0,            0,            1,              0);
	invK.data[3] = make_float4(0,            0,            0,              1);
	return invK;
}
inline float4 operator*(const Matrix4 & M, const float4 & v) {
	return make_float4(my_dot(M.data[0], v),
                     my_dot(M.data[1], v),
                     my_dot(M.data[2], v),
                     my_dot(M.data[3], v));
}

inline Matrix4 inverse(const Matrix4 & A) {
  static TooN::Matrix<4, 4, float> I = TooN::Identity;
//  TooN::Matrix<4, 4, float> temp = TooN::wrapMatrix<4, 4>(&A.data[0].x());
  TooN::Matrix<4, 4, float> temp =
    TooN::wrapMatrix<4, 4>(reinterpret_cast<const float *>(&A));
  Matrix4 R;
//This approach gives a segmentation fault:
//TooN::wrapMatrix<4, 4>(&R.data[0].x()) = TooN::gaussian_elimination(temp, I);
  TooN::wrapMatrix<4, 4>(reinterpret_cast<float *>(&R)) = TooN::gaussian_elimination(temp, I);
  return R;
}

inline Matrix4 operator*(const Matrix4 & A, const Matrix4 & B) {
	Matrix4 R;
  TooN::wrapMatrix<4, 4>(reinterpret_cast<float *>(&R)) =
    TooN::wrapMatrix<4, 4>(reinterpret_cast<const float *>(&A)) *
    TooN::wrapMatrix<4, 4>(reinterpret_cast<const float *>(&B));
  return R;
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

template<typename P>
inline Matrix4 toMatrix4(const TooN::SE3<P> & p) {
	const TooN::Matrix<4, 4, float> I = TooN::Identity;
	Matrix4 R;
//	TooN::wrapMatrix<4, 4>(&R.data[0].x()) = p * I;
	TooN::wrapMatrix<4, 4>(reinterpret_cast<float *>(&R)) = p * I;
	return R;
}

static const float epsilon = 0.0000001;

inline void compareTrackData(std::string str, TrackData* l, TrackData * r,
		uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i].error - r[i].error) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.error =  " << l[i].error << std::endl;
			std::cout << "r.error =  " << r[i].error << std::endl;
		}

		if (std::abs(l[i].result - r[i].result) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.result =  " << l[i].result << std::endl;
			std::cout << "r.result =  " << r[i].result << std::endl;
		}

	}
}

inline void compareFloat(std::string str, float* l, float * r, uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i] - r[i]) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l =  " << l[i] << std::endl;
			std::cout << "r =  " << r[i] << std::endl;
		}
	}
}
inline void compareFloat3(std::string str, float3* l, float3 * r, uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i].x() - r[i].x()) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.x() =  " << l[i].x() << std::endl;
			std::cout << "r.x() =  " << r[i].x() << std::endl;
		}
		if (std::abs(l[i].y() - r[i].y()) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.y() =  " << l[i].y() << std::endl;
			std::cout << "r.y() =  " << r[i].y() << std::endl;
		}
		if (std::abs(l[i].z() - r[i].z()) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.z() =  " << l[i].z() << std::endl;
			std::cout << "r.z() =  " << r[i].z() << std::endl;
		}
	}
}

inline void compareFloat4(std::string str, float4* l, float4 * r, uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i].x() - r[i].x()) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.x() =  " << l[i].x() << std::endl;
			std::cout << "r.x() =  " << r[i].x() << std::endl;
		}
		if (std::abs(l[i].y() - r[i].y()) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.y() =  " << l[i].y() << std::endl;
			std::cout << "r.y() =  " << r[i].y() << std::endl;
		}
		if (std::abs(l[i].z() - r[i].z()) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.z() =  " << l[i].z() << std::endl;
			std::cout << "r.z() =  " << r[i].z() << std::endl;
		}
		if (std::abs(l[i].w() - r[i].w()) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.w() =  " << l[i].w() << std::endl;
			std::cout << "r.w() =  " << r[i].w() << std::endl;
		}
	}
}

inline void compareMatrix4(std::string str, Matrix4 l, Matrix4 r) {
	compareFloat4(str, l.data, r.data, 4);
}

inline void printMatrix4(std::string str, Matrix4 l) {
	std::cout << "printMatrix4 : " << str << std::endl;
	for (int i = 0; i < 4; i++) {
		std::cout << "  [" << l.data[i].x() << "," << l.data[i].y() << ","
				<< l.data[i].z() << "," << l.data[i].w() << "]" << std::endl;
	}
}
inline void compareNormal(std::string str, float3* l, float3 * r, uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i].x() - r[i].x()) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.x() =  " << l[i].x() << std::endl;
			std::cout << "r.x() =  " << r[i].x() << std::endl;
		} else if (r[i].x() != INVALID) {
			if (std::abs(l[i].y() - r[i].y()) > epsilon) {
				std::cout << "Error into " << str << " at " << i << std::endl;
				std::cout << "l.y() =  " << l[i].y() << std::endl;
				std::cout << "r.y() =  " << r[i].y() << std::endl;
			}
			if (std::abs(l[i].z() - r[i].z()) > epsilon) {
				std::cout << "Error into " << str << " at " << i << std::endl;
				std::cout << "l.z() =  " << l[i].z() << std::endl;
				std::cout << "r.z() =  " << r[i].z() << std::endl;
			}
		}
	}
}

template<typename T>
void writefile(std::string prefix, int idx, T * data, uint size) {

	std::string filename = prefix + NumberToString(idx);
	FILE* pFile = fopen(filename.c_str(), "wb");

	if (!pFile) {
		std::cout << "File opening failed : " << filename << std::endl;
		exit(1);
	}

	size_t write_cnt = fwrite(data, sizeof(T), size, pFile);

	std::cout << "File " << filename << " of size " << write_cnt << std::endl;

	fclose(pFile);
}

template<typename T>
void writefile(std::string prefix, int idx, T * data, uint2 size) {
	writefile(prefix, idx, data, size.x() * size.y());
}
inline
void writeposfile(std::string prefix, int idx, Matrix4 m, uint) {

	writefile("BINARY_" + prefix, idx, m.data, 4);

	std::string filename = prefix + NumberToString(idx);
	std::ofstream pFile;
	pFile.open(filename.c_str());

	if (pFile.fail()) {
		std::cout << "File opening failed : " << filename << std::endl;
		exit(1);
	}

	pFile << m.data[0].x() << " " << m.data[0].y() << " " << m.data[0].z() << " "
			<< m.data[0].w() << std::endl;
	pFile << m.data[1].x() << " " << m.data[1].y() << " " << m.data[1].z() << " "
			<< m.data[1].w() << std::endl;
	pFile << m.data[2].x() << " " << m.data[2].y() << " " << m.data[2].z() << " "
			<< m.data[2].w() << std::endl;
	pFile << m.data[3].x() << " " << m.data[3].y() << " " << m.data[3].z() << " "
			<< m.data[3].w() << std::endl;

	std::cout << "Pose File " << filename << std::endl;

	pFile.close();
}
template <typename T>
inline
void writeVolume(std::string filename, Volume<T> v) {

	std::ofstream fDumpFile;
	fDumpFile.open(filename.c_str(), std::ios::out | std::ios::binary);

	if (fDumpFile == NULL) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}

	// Retrieve the volumetric representation data
	short2 *hostData = (short2 *) v.data;

	// Dump on file without the y component of the short2 variable
	for (unsigned int i = 0; i < v.size.x() * v.size.y() * v.size.z(); i++) {
		fDumpFile.write((char *) (hostData + i), sizeof(short));
	}

	fDumpFile.close();
}

#endif
