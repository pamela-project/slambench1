/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

/************** TYPES ***************/

#define INVALID -2 

typedef struct sVolume {
	uint3 size;
	float3 dim;
	__global short2 * data;
} Volume;

typedef struct sTrackData {
	int result;
	float error;
	float J[6];
} TrackData;

typedef struct sMatrix4 {
	float4 data[4];
} Matrix4;

/************** FUNCTIONS ***************/

inline float sq(float r) {
	return r * r;
}

inline float3 Mat4TimeFloat3(Matrix4 M, float3 v) {
	return (float3)(
			dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), v)
					+ M.data[0].w,
			dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), v)
					+ M.data[1].w,
			dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), v)
					+ M.data[2].w);
}

inline void setVolume(Volume v, uint3 pos, float2 d) {
	v.data[pos.x + pos.y * v.size.x + pos.z * v.size.x * v.size.y] = (short2)(
			d.x * 32766.0f, d.y);
}

inline float3 posVolume(const Volume v, const uint3 p) {
	return (float3)((p.x + 0.5f) * v.dim.x / v.size.x,
			(p.y + 0.5f) * v.dim.y / v.size.y,
			(p.z + 0.5f) * v.dim.z / v.size.z);
}

inline float2 getVolume(const Volume v, const uint3 pos) {
	const short2 d = v.data[pos.x + pos.y * v.size.x
			+ pos.z * v.size.x * v.size.y];
	return (float2)(d.x * 0.00003051944088f, d.y); //  / 32766.0f
}

inline float vs(const uint3 pos, const Volume v) {
	return v.data[pos.x + pos.y * v.size.x + pos.z * v.size.x * v.size.y].x;
}

inline float interp(const float3 pos, const Volume v) {
	const float3 scaled_pos = (float3)((pos.x * v.size.x / v.dim.x) - 0.5f,
			(pos.y * v.size.y / v.dim.y) - 0.5f,
			(pos.z * v.size.z / v.dim.z) - 0.5f);
	float3 basef = (float3)(0);
	const int3 base = convert_int3(floor(scaled_pos));
	const float3 factor = (float3)(fract(scaled_pos, (float3 *) &basef));
	const int3 lower = max(base, (int3)(0));
	const int3 upper = min(base + (int3)(1), convert_int3(v.size) - (int3)(1));
	return (((vs((uint3)(lower.x, lower.y, lower.z), v) * (1 - factor.x)
			+ vs((uint3)(upper.x, lower.y, lower.z), v) * factor.x)
			* (1 - factor.y)
			+ (vs((uint3)(lower.x, upper.y, lower.z), v) * (1 - factor.x)
					+ vs((uint3)(upper.x, upper.y, lower.z), v) * factor.x)
					* factor.y) * (1 - factor.z)
			+ ((vs((uint3)(lower.x, lower.y, upper.z), v) * (1 - factor.x)
					+ vs((uint3)(upper.x, lower.y, upper.z), v) * factor.x)
					* (1 - factor.y)
					+ (vs((uint3)(lower.x, upper.y, upper.z), v)
							* (1 - factor.x)
							+ vs((uint3)(upper.x, upper.y, upper.z), v)
									* factor.x) * factor.y) * factor.z)
			* 0.00003051944088f;
}

inline float3 grad(float3 pos, const Volume v) {
	const float3 scaled_pos = (float3)((pos.x * v.size.x / v.dim.x) - 0.5f,
			(pos.y * v.size.y / v.dim.y) - 0.5f,
			(pos.z * v.size.z / v.dim.z) - 0.5f);
	const int3 base = (int3)(floor(scaled_pos.x), floor(scaled_pos.y),
			floor(scaled_pos.z));
	const float3 basef = (float3)(0);
	const float3 factor = (float3) fract(scaled_pos, (float3 *) &basef);
	const int3 lower_lower = max(base - (int3)(1), (int3)(0));
	const int3 lower_upper = max(base, (int3)(0));
	const int3 upper_lower = min(base + (int3)(1),
			convert_int3(v.size) - (int3)(1));
	const int3 upper_upper = min(base + (int3)(2),
			convert_int3(v.size) - (int3)(1));
	const int3 lower = lower_upper;
	const int3 upper = upper_lower;

	float3 gradient;

	gradient.x = (((vs((uint3)(upper_lower.x, lower.y, lower.z), v)
			- vs((uint3)(lower_lower.x, lower.y, lower.z), v)) * (1 - factor.x)
			+ (vs((uint3)(upper_upper.x, lower.y, lower.z), v)
					- vs((uint3)(lower_upper.x, lower.y, lower.z), v))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(upper_lower.x, upper.y, lower.z), v)
					- vs((uint3)(lower_lower.x, upper.y, lower.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper_upper.x, upper.y, lower.z), v)
							- vs((uint3)(lower_upper.x, upper.y, lower.z), v))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(upper_lower.x, lower.y, upper.z), v)
					- vs((uint3)(lower_lower.x, lower.y, upper.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper_upper.x, lower.y, upper.z), v)
							- vs((uint3)(lower_upper.x, lower.y, upper.z), v))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(upper_lower.x, upper.y, upper.z), v)
							- vs((uint3)(lower_lower.x, upper.y, upper.z), v))
							* (1 - factor.x)
							+ (vs((uint3)(upper_upper.x, upper.y, upper.z), v)
									- vs(
											(uint3)(lower_upper.x, upper.y,
													upper.z), v)) * factor.x)
							* factor.y) * factor.z;

	gradient.y = (((vs((uint3)(lower.x, upper_lower.y, lower.z), v)
			- vs((uint3)(lower.x, lower_lower.y, lower.z), v)) * (1 - factor.x)
			+ (vs((uint3)(upper.x, upper_lower.y, lower.z), v)
					- vs((uint3)(upper.x, lower_lower.y, lower.z), v))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(lower.x, upper_upper.y, lower.z), v)
					- vs((uint3)(lower.x, lower_upper.y, lower.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper_upper.y, lower.z), v)
							- vs((uint3)(upper.x, lower_upper.y, lower.z), v))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(lower.x, upper_lower.y, upper.z), v)
					- vs((uint3)(lower.x, lower_lower.y, upper.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper_lower.y, upper.z), v)
							- vs((uint3)(upper.x, lower_lower.y, upper.z), v))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(lower.x, upper_upper.y, upper.z), v)
							- vs((uint3)(lower.x, lower_upper.y, upper.z), v))
							* (1 - factor.x)
							+ (vs((uint3)(upper.x, upper_upper.y, upper.z), v)
									- vs(
											(uint3)(upper.x, lower_upper.y,
													upper.z), v)) * factor.x)
							* factor.y) * factor.z;

	gradient.z = (((vs((uint3)(lower.x, lower.y, upper_lower.z), v)
			- vs((uint3)(lower.x, lower.y, lower_lower.z), v)) * (1 - factor.x)
			+ (vs((uint3)(upper.x, lower.y, upper_lower.z), v)
					- vs((uint3)(upper.x, lower.y, lower_lower.z), v))
					* factor.x) * (1 - factor.y)
			+ ((vs((uint3)(lower.x, upper.y, upper_lower.z), v)
					- vs((uint3)(lower.x, upper.y, lower_lower.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, upper.y, upper_lower.z), v)
							- vs((uint3)(upper.x, upper.y, lower_lower.z), v))
							* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs((uint3)(lower.x, lower.y, upper_upper.z), v)
					- vs((uint3)(lower.x, lower.y, lower_upper.z), v))
					* (1 - factor.x)
					+ (vs((uint3)(upper.x, lower.y, upper_upper.z), v)
							- vs((uint3)(upper.x, lower.y, lower_upper.z), v))
							* factor.x) * (1 - factor.y)
					+ ((vs((uint3)(lower.x, upper.y, upper_upper.z), v)
							- vs((uint3)(lower.x, upper.y, lower_upper.z), v))
							* (1 - factor.x)
							+ (vs((uint3)(upper.x, upper.y, upper_upper.z), v)
									- vs(
											(uint3)(upper.x, upper.y,
													lower_upper.z), v))
									* factor.x) * factor.y) * factor.z;

	return gradient
			* (float3)(v.dim.x / v.size.x, v.dim.y / v.size.y,
					v.dim.z / v.size.z) * (0.5f * 0.00003051944088f);
}

inline float3 get_translation(const Matrix4 view) {
	return (float3)(view.data[0].w, view.data[1].w, view.data[2].w);
}

inline float3 myrotate(const Matrix4 M, const float3 v) {
	return (float3)(dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), v),
			dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), v),
			dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), v));
}

float4 raycast(const Volume v, const uint2 pos, const Matrix4 view,
		const float nearPlane, const float farPlane, const float step,
		const float largestep) {

	const float3 origin = get_translation(view);
	const float3 direction = myrotate(view, (float3)(pos.x, pos.y, 1.f));

	// intersect ray with a box
	//
	// www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
	// compute intersection of ray with all six bbox planes
	const float3 invR = (float3)(1.0f) / direction;
	const float3 tbot = (float3) - 1 * invR * origin;
	const float3 ttop = invR * (v.dim - origin);

	// re-order intersections to find smallest and largest on each axis
	const float3 tmin = fmin(ttop, tbot);
	const float3 tmax = fmax(ttop, tbot);

	// find the largest tmin and the smallest tmax
	const float largest_tmin = fmax(fmax(tmin.x, tmin.y), fmax(tmin.x, tmin.z));
	const float smallest_tmax = fmin(fmin(tmax.x, tmax.y),
			fmin(tmax.x, tmax.z));

	// check against near and far plane
	const float tnear = fmax(largest_tmin, nearPlane);
	const float tfar = fmin(smallest_tmax, farPlane);

	if (tnear < tfar) {
		// first walk with largesteps until we found a hit
		float t = tnear;
		float stepsize = largestep;
		float f_t = interp(origin + direction * t, v);
		float f_tt = 0;
		if (f_t > 0) { // ups, if we were already in it, then don't render anything here
			for (; t < tfar; t += stepsize) {
				f_tt = interp(origin + direction * t, v);
				if (f_tt < 0)                  // got it, jump out of inner loop
					break;
				if (f_tt < 0.8f)               // coming closer, reduce stepsize
					stepsize = step;
				f_t = f_tt;
			}
			if (f_tt < 0) {           // got it, calculate accurate intersection
				t = t + stepsize * f_tt / (f_t - f_tt);
				return (float4)(origin + direction * t, t);
			}
		}
	}

	return (float4)(0);
}

/************** KERNELS ***************/

__kernel void renderNormalKernel( const __global uchar * in,
		__global float * out ) {

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);

	float3 n;
	const uchar3 i = vload3(posx + sizex * posy,in);

	n.x = i.x;
	n.y = i.y;
	n.z = i.z;

	if(n.x == -2) {
		vstore3((float3) (0,0,0),posx + sizex * posy,out);
	} else {
		n = normalize(n);
		vstore3((float3) (n.x*128 + 128,
						n.y*128+128, n.z*128+128), posx + sizex * posy,out);
	}

}

__kernel void renderDepthKernel( __global uchar4 * out,
		__global float * depth,
		const float nearPlane,
		const float farPlane ) {

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);
	float d= depth[posx + sizex * posy];
	if(d < nearPlane)
	vstore4((uchar4)(255, 255, 255, 0), posx + sizex * posy, (__global uchar*)out); // The forth value in uchar4 is padding for memory alignement and so it is for following uchar4 
	else {
		if(d > farPlane)
		vstore4((uchar4)(0, 0, 0, 0), posx + sizex * posy, (__global uchar*)out);
		else {
			float h =(d - nearPlane) / (farPlane - nearPlane);
			h *= 6.0f;
			const int sextant = (int)h;
			const float fract = h - sextant;
			const float mid1 = 0.25f + (0.5f*fract);
			const float mid2 = 0.75f - (0.5f*fract);
			switch (sextant)
			{
				case 0: vstore4((uchar4)(191, 255*mid1, 64, 0), posx + sizex * posy, (__global uchar*)out); break;
				case 1: vstore4((uchar4)(255*mid2, 191, 64, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 2: vstore4((uchar4)(64, 191, 255*mid1, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 3: vstore4((uchar4)(64, 255*mid2, 191, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 4: vstore4((uchar4)(255*mid1, 64, 191, 0),posx + sizex * posy ,(__global uchar*)out); break;
				case 5: vstore4((uchar4)(191, 64, 255*mid2, 0),posx + sizex * posy ,(__global uchar*)out); break;
			}
		}
	}
}

__kernel void renderTrackKernel( __global uchar3 * out,
		__global const TrackData * data ) {

	const int posx = get_global_id(0);
	const int posy = get_global_id(1);
	const int sizex = get_global_size(0);

	switch(data[posx + sizex * posy].result) {
		// The forth value in uchar4 is padding for memory alignement and so it is for following uchar4
		case  1: vstore4((uchar4)(128, 128, 128, 0), posx + sizex * posy, (__global uchar*)out); break; // ok	 GREY
		case -1: vstore4((uchar4)(000, 000, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // no input BLACK
		case -2: vstore4((uchar4)(255, 000, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // not in image RED
		case -3: vstore4((uchar4)(000, 255, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // no correspondence GREEN
		case -4: vstore4((uchar4)(000, 000, 255, 0), posx + sizex * posy, (__global uchar*)out); break; // too far away BLUE
		case -5: vstore4((uchar4)(255, 255, 000, 0), posx + sizex * posy, (__global uchar*)out); break; // wrong normal YELLOW
		default: vstore4((uchar4)(255, 128, 128, 0), posx + sizex * posy, (__global uchar*)out); return;
	}
}

__kernel void bilateralFilterKernel( __global float * out,
		const __global float * in,
		const __global float * gaussian,
		const float e_d,
		const int r ) {

	const uint2 pos = (uint2) (get_global_id(0),get_global_id(1));
	const uint2 size = (uint2) (get_global_size(0),get_global_size(1));

	const float center = in[pos.x + size.x * pos.y];

	if ( center == 0 ) {
		out[pos.x + size.x * pos.y] = 0;
		return;
	}

	float sum = 0.0f;
	float t = 0.0f;
	// FIXME : sum and t diverge too much from cpp version
	for(int i = -r; i <= r; ++i) {
		for(int j = -r; j <= r; ++j) {
			const uint2 curPos = (uint2)(clamp(pos.x + i, 0u, size.x-1), clamp(pos.y + j, 0u, size.y-1));
			const float curPix = in[curPos.x + curPos.y * size.x];
			if(curPix > 0) {
				const float mod = sq(curPix - center);
				const float factor = gaussian[i + r] * gaussian[j + r] * exp(-mod / (2 * e_d * e_d));
				t += factor * curPix;
				sum += factor;
			} else {
				//std::cerr << "ERROR BILATERAL " <<pos.x+i<< " "<<pos.y+j<< " " <<curPix<<" \n";
			}
		}
	}
	out[pos.x + size.x * pos.y] = t / sum;

}

__kernel void renderVolumeKernel( __global uchar * render,
		__global short2 * v_data,
		const uint3 v_size,
		const float3 v_dim,
		const Matrix4 view,
		const float nearPlane,
		const float farPlane,
		const float step,
		const float largestep,
		const float3 light,
		const float3 ambient) {

	const Volume v = {v_size, v_dim,v_data};

	const uint2 pos = (uint2) (get_global_id(0),get_global_id(1));
	const int sizex = get_global_size(0);

	float4 hit = raycast( v, pos, view, nearPlane, farPlane,step, largestep);

	if(hit.w > 0) {
		const float3 test = as_float3(hit);
		float3 surfNorm = grad(test,v);
		if(length(surfNorm) > 0) {
			const float3 diff = normalize(light - test);
			const float dir = fmax(dot(normalize(surfNorm), diff), 0.f);
			const float3 col = clamp((float3)(dir) + ambient, 0.f, 1.f) * (float3) 255;
			vstore4((uchar4)(col.x, col.y, col.z, 0), pos.x + sizex * pos.y, render); // The forth value in uchar4 is padding for memory alignement and so it is for following uchar4 
		} else {
			vstore4((uchar4)(0, 0, 0, 0), pos.x + sizex * pos.y, render);
		}
	} else {
		vstore4((uchar4)(0, 0, 0, 0), pos.x + sizex * pos.y, render);
	}

}

/************** KFUSION KERNELS ***************/

__kernel void raycastKernel( __global float * pos3D,  //float3
		__global float * normal,//float3
		__global short2 * v_data,
		const uint3 v_size,
		const float3 v_dim,
		const Matrix4 view,
		const float nearPlane,
		const float farPlane,
		const float step,
		const float largestep ) {

	const Volume volume = {v_size, v_dim,v_data};

	const uint2 pos = (uint2) (get_global_id(0),get_global_id(1));
	const int sizex = get_global_size(0);

	const float4 hit = raycast( volume, pos, view, nearPlane, farPlane, step, largestep );
	const float3 test = as_float3(hit);

	if(hit.w > 0.0f ) {
		vstore3(test,pos.x + sizex * pos.y,pos3D);
		float3 surfNorm = grad(test,volume);
		if(length(surfNorm) == 0) {
			//float3 n =  (INVALID,0,0);//vload3(pos.x + sizex * pos.y,normal);
			//n.x=INVALID;
			vstore3((float3)(INVALID,INVALID,INVALID),pos.x + sizex * pos.y,normal);
		} else {
			vstore3(normalize(surfNorm),pos.x + sizex * pos.y,normal);
		}
	} else {
		vstore3((float3)(0),pos.x + sizex * pos.y,pos3D);
		vstore3((float3)(INVALID, INVALID, INVALID),pos.x + sizex * pos.y,normal);
	}
}

__kernel void integrateKernel (
		__global short2 * v_data,
		const uint3 v_size,
		const float3 v_dim,
		__global const float * depth,
		const uint2 depthSize,
		const Matrix4 invTrack,
		const Matrix4 K,
		const float mu,
		const float maxweight ,
		const float3 delta ,
		const float3 cameraDelta
) {

	Volume vol; vol.data = v_data; vol.size = v_size; vol.dim = v_dim;

	uint3 pix = (uint3) (get_global_id(0),get_global_id(1),0);
	const int sizex = get_global_size(0);

	float3 pos = Mat4TimeFloat3 (invTrack , posVolume(vol,pix));
	float3 cameraX = Mat4TimeFloat3 ( K , pos);

	for(pix.z = 0; pix.z < vol.size.z; ++pix.z, pos += delta, cameraX += cameraDelta) {
		if(pos.z < 0.0001f) // some near plane constraint
		continue;
		const float2 pixel = (float2) (cameraX.x/cameraX.z + 0.5f, cameraX.y/cameraX.z + 0.5f);

		if(pixel.x < 0 || pixel.x > depthSize.x-1 || pixel.y < 0 || pixel.y > depthSize.y-1)
		continue;
		const uint2 px = (uint2)(pixel.x, pixel.y);
		float depthpx = depth[px.x + depthSize.x * px.y];

		if(depthpx == 0) continue;
		const float diff = ((depthpx) - cameraX.z) * sqrt(1+sq(pos.x/pos.z) + sq(pos.y/pos.z));

		if(diff > -mu) {
			const float sdf = fmin(1.f, diff/mu);
			float2 data = getVolume(vol,pix);
			data.x = clamp((data.y*data.x + sdf)/(data.y + 1), -1.f, 1.f);
			data.y = fmin(data.y+1, maxweight);
			setVolume(vol,pix, data);
		}
	}

}

// inVertex iterate
__kernel void trackKernel (
		__global TrackData * output,
		const uint2 outputSize,
		__global const float * inVertex,// float3
		const uint2 inVertexSize,
		__global const float * inNormal,// float3
		const uint2 inNormalSize,
		__global const float * refVertex,// float3
		const uint2 refVertexSize,
		__global const float * refNormal,// float3
		const uint2 refNormalSize,
		const Matrix4 Ttrack,
		const Matrix4 view,
		const float dist_threshold,
		const float normal_threshold
) {

	const uint2 pixel = (uint2)(get_global_id(0),get_global_id(1));

	if(pixel.x >= inVertexSize.x || pixel.y >= inVertexSize.y ) {return;}

	float3 inNormalPixel = vload3(pixel.x + inNormalSize.x * pixel.y,inNormal);

	if(inNormalPixel.x == INVALID ) {
		output[pixel.x + outputSize.x * pixel.y].result = -1;
		return;
	}

	float3 inVertexPixel = vload3(pixel.x + inVertexSize.x * pixel.y,inVertex);
	const float3 projectedVertex = Mat4TimeFloat3 (Ttrack , inVertexPixel);
	const float3 projectedPos = Mat4TimeFloat3 ( view , projectedVertex);
	const float2 projPixel = (float2) ( projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

	if(projPixel.x < 0 || projPixel.x > refVertexSize.x-1 || projPixel.y < 0 || projPixel.y > refVertexSize.y-1 ) {
		output[pixel.x + outputSize.x * pixel.y].result = -2;
		return;
	}

	const uint2 refPixel = (uint2) (projPixel.x, projPixel.y);
	const float3 referenceNormal = vload3(refPixel.x + refNormalSize.x * refPixel.y,refNormal);

	if(referenceNormal.x == INVALID) {
		output[pixel.x + outputSize.x * pixel.y].result = -3;
		return;
	}

	const float3 diff = vload3(refPixel.x + refVertexSize.x * refPixel.y,refVertex) - projectedVertex;
	const float3 projectedNormal = myrotate(Ttrack, inNormalPixel);

	if(length(diff) > dist_threshold ) {
		output[pixel.x + outputSize.x * pixel.y].result = -4;
		return;
	}
	if(dot(projectedNormal, referenceNormal) < normal_threshold) {
		output[pixel.x + outputSize.x * pixel.y] .result = -5;
		return;
	}

	output[pixel.x + outputSize.x * pixel.y].result = 1;
	output[pixel.x + outputSize.x * pixel.y].error = dot(referenceNormal, diff);

	vstore3(referenceNormal,0,(output[pixel.x + outputSize.x * pixel.y].J));
	vstore3(cross(projectedVertex, referenceNormal),1,(output[pixel.x + outputSize.x * pixel.y].J));

}

__kernel void reduceKernel (
		__global float * out,
		__global const TrackData * J,
		const uint2 JSize,
		const uint2 size,
		__local float * S
) {

	uint blockIdx = get_group_id(0);
	uint blockDim = get_local_size(0);
	uint threadIdx = get_local_id(0);
	uint gridDim = get_num_groups(0);

	const uint sline = threadIdx;

	float sums[32];
	float * jtj = sums + 7;
	float * info = sums + 28;

	for(uint i = 0; i < 32; ++i)
	sums[i] = 0.0f;

	for(uint y = blockIdx; y < size.y; y += gridDim) {
		for(uint x = sline; x < size.x; x += blockDim ) {
			const TrackData row = J[x + y * JSize.x];
			if(row.result < 1) {
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

	for(int i = 0; i < 32; ++i) // copy over to shared memory
	S[sline * 32 + i] = sums[i];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(sline < 32) { // sum up columns and copy to global memory in the final 32 threads
		for(unsigned i = 1; i < blockDim; ++i)
		S[sline] += S[i * 32 + sline];
		out[sline+blockIdx*32] = S[sline];
	}
}

__kernel void depth2vertexKernel( __global float * vertex, // float3
		const uint2 vertexSize ,
		const __global float * depth,
		const uint2 depthSize ,
		const Matrix4 invK ) {

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	float3 vert = (float3)(get_global_id(0),get_global_id(1),1.0f);

	if(pixel.x >= depthSize.x || pixel.y >= depthSize.y ) {
		return;
	}

	float3 res = (float3) (0);

	if(depth[pixel.x + depthSize.x * pixel.y] > 0) {
		res = depth[pixel.x + depthSize.x * pixel.y] * (myrotate(invK, (float3)(pixel.x, pixel.y, 1.f)));
	}

	vstore3(res, pixel.x + vertexSize.x * pixel.y,vertex); 	// vertex[pixel] =

}

__kernel void vertex2normalKernel( __global float * normal,    // float3
		const uint2 normalSize,
		const __global float * vertex ,
		const uint2 vertexSize ) {  // float3

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));

	if(pixel.x >= vertexSize.x || pixel.y >= vertexSize.y )
	return;

	//const float3 left = vertex[(uint2)(max(int(pixel.x)-1,0), pixel.y)];
	//const float3 right = vertex[(uint2)(min(pixel.x+1,vertex.size.x-1), pixel.y)];
	//const float3 up = vertex[(uint2)(pixel.x, max(int(pixel.y)-1,0))];
	//const float3 down = vertex[(uint2)(pixel.x, min(pixel.y+1,vertex.size.y-1))];

	uint2 vleft = (uint2)(max((int)(pixel.x)-1,0), pixel.y);
	uint2 vright = (uint2)(min(pixel.x+1,vertexSize.x-1), pixel.y);
	uint2 vup = (uint2)(pixel.x, max((int)(pixel.y)-1,0));
	uint2 vdown = (uint2)(pixel.x, min(pixel.y+1,vertexSize.y-1));

	const float3 left = vload3(vleft.x + vertexSize.x * vleft.y,vertex);
	const float3 right = vload3(vright.x + vertexSize.x * vright.y,vertex);
	const float3 up = vload3(vup.x + vertexSize.x * vup.y,vertex);
	const float3 down = vload3(vdown.x + vertexSize.x * vdown.y,vertex);
	/*
	 unsigned long int val =  0 ;
	 val = max(((int) pixel.x)-1,0) + vertexSize.x * pixel.y;
	 const float3 left   = vload3(   val,vertex);

	 val =  min(pixel.x+1,vertexSize.x-1)                  + vertexSize.x *     pixel.y;
	 const float3 right  = vload3(    val     ,vertex);
	 val =   pixel.x                        + vertexSize.x *     max(((int) pixel.y)-1,0)  ;
	 const float3 up     = vload3(  val ,vertex);
	 val =  pixel.x                       + vertexSize.x *   min(pixel.y+1,vertexSize.y-1)   ;
	 const float3 down   = vload3(  val   ,vertex);
	 */
	if(left.z == 0 || right.z == 0|| up.z ==0 || down.z == 0) {
		//float3 n = vload3(pixel.x + normalSize.x * pixel.y,normal);
		//n.x=INVALID;
		vstore3((float3)(INVALID,INVALID,INVALID),pixel.x + normalSize.x * pixel.y,normal);
		return;
	}
	const float3 dxv = right - left;
	const float3 dyv = down - up;
	vstore3((float3) normalize(cross(dyv, dxv)), pixel.x + pixel.y * normalSize.x, normal );

}

__kernel void mm2metersKernel(
		__global float * depth,
		const uint2 depthSize ,
		const __global ushort * in ,
		const uint2 inSize ,
		const int ratio ) {
	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	depth[pixel.x + depthSize.x * pixel.y] = in[pixel.x * ratio + inSize.x * pixel.y * ratio] / 1000.0f;
}

__kernel void initVolumeKernel(__global short2 * data) {

	uint x = get_global_id(0);
	uint y = get_global_id(1);
	uint z = get_global_id(2);
	uint3 size = (uint3) (get_global_size(0),get_global_size(1),get_global_size(2));
	float2 d = (float2) (1.0f,0.0f);

	data[x + y * size.x + z * size.x * size.y] = (short2) (d.x * 32766.0f, d.y);

}

__kernel void halfSampleRobustImageKernel(__global float * out,
		__global const float * in,
		const uint2 inSize,
		const float e_d,
		const int r) {

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	uint2 outSize = inSize / 2;

	const uint2 centerPixel = 2 * pixel;

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[centerPixel.x + centerPixel.y * inSize.x];
	for(int i = -r + 1; i <= r; ++i) {
		for(int j = -r + 1; j <= r; ++j) {
			int2 from = (int2)(clamp((int2)(centerPixel.x + j, centerPixel.y + i), (int2)(0), (int2)(inSize.x - 1, inSize.y - 1)));
			float current = in[from.x + from.y * inSize.x];
			if(fabs(current - center) < e_d) {
				sum += 1.0f;
				t += current;
			}
		}
	}
	out[pixel.x + pixel.y * outSize.x] = t / sum;

}

