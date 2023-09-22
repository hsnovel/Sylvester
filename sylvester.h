// Sylvester - optimized linear math library
// https://github.com/hsnovel/Sylvester
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// MIT License
// Copyright (c) Çağan Korkmaz <cagankorkmaz35@gmail.com>
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*   Sylvester
 *
 *   ===========================================================================
 *
 *                                    IMPORTANT
 *
 *   DO NOT ACCESS VARIABLE OR FUNCTION THAT START WITH UNDERSCORE. THEY ARE
 *   INTERNAL ONLY AND CHANGING IT MIGHT BREAK THE LIBRARY OR CAUSE UNEXPECTED
 *   BEHAVIOURS !
 *
 *   ===========================================================================
 *
 */

#ifndef SYLVESTER_H
#define SYLVESTER_H

#if defined(SYL_ENABLE_AVX)
#include <immintrin.h>
#elif defined(SYL_ENABLE_SSE4)
#include <smmintrin.h>
#endif

#include <string.h>
#include <math.h>
#include <stdbool.h>

#if defined(__clang__)
#	define _SYL_SET_SPEC_ALIGN(x) __attribute__((aligned(x)))

#elif defined(__GNUC__) || defined(__GNUG__)
#	define _SYL_SET_SPEC_ALIGN(x) __attribute__((aligned(x)))

#elif defined(_MSC_VER)
#	define _SYL_SET_SPEC_ALIGN(x) __declspec(align(x))
#else
#	define _SYL_SET_SPEC_ALIGN(x)
#endif

#define SYL_INLINE

/* Doing it the normal way fucks up emacs indentation*/
#define _SYL_CPP_EXTER_START extern "C" {
#define _SYL_CPP_EXTERN_END }

#ifdef __cplusplus
_SYL_CPP_EXTER_START
#endif

typedef union svec2
{
	struct { float x, y; };
	struct { float s; float t; };
	float e[2];
#if defined(SYL_ENABLE_SSE4)
	__m64 v;
#endif
} _SYL_SET_SPEC_ALIGN(8) svec2;

typedef union svec3
{
	struct { float x; float y; float z; };
	struct { float r; float g; float b; };
	struct { float s; float t; float p; };
	float e[3];
} svec3;

typedef union svec4
{
	struct { float x; float y; float z; float w; };
	struct { float r; float g; float b; float a; };
	struct { float s; float t; float m; float q; };
	float e[4];
#if defined(SYL_ENABLE_SSE4)
	__m128 v;
#endif
} _SYL_SET_SPEC_ALIGN(16) svec4;

/* We use column-major matricies */
typedef union smat4
{
	struct
	{
		float m00, m01, m02, m03;
		float m10, m11, m12, m13;
		float m20, m21, m22, m23;
		float m30, m31, m32, m33;
	};
	float e[16];
	float e2[4][4];
	svec4 v4d[4];
#if defined(SYL_ENABLE_SSE4)
	__m128 v[4];
#endif
#if defined(SYL_ENABLE_AVX)
	__m256 v2[2];
#endif
} _SYL_SET_SPEC_ALIGN(16) smat4;

SYL_INLINE float s_radian_to_degree(float Radian);
SYL_INLINE float s_degree_to_radian(float Degree);
SYL_INLINE float s_roundf(float A);
SYL_INLINE double s_roundd(double A);
SYL_INLINE float s_ceilf(float A);
SYL_INLINE double s_ceild(double A);
SYL_INLINE float s_floorf(float A);
SYL_INLINE double s_floord(double A);
SYL_INLINE svec4 s_bgra_unpack(int Color);
SYL_INLINE unsigned int s_bgra_pack(svec4 Color);
SYL_INLINE svec4 s_rgba_unpack(unsigned int Color);
SYL_INLINE unsigned int s_rgba_pack(svec4 Color);
SYL_INLINE float s_clampf(float Value, float Min, float Max);
SYL_INLINE float s_clampd(float Value, float Min, float Max);
SYL_INLINE int s_clampi(int Value, int Min, int Max);
SYL_INLINE float s_clamp01f(float Value);
SYL_INLINE double s_clamp01d(double Value);
SYL_INLINE float s_clamp_above_zero(float Value);
SYL_INLINE float s_clamp_below_zero(float Value);
SYL_INLINE bool s_is_in_range(float Value, float Min, float Max);
SYL_INLINE float s_lerp(float A, float t, float B);
SYL_INLINE float s_square(float x);
SYL_INLINE float s_abs(float x);
SYL_INLINE float s_pythagorean(float x, float y);
SYL_INLINE float s_maxf(float x, float y);
SYL_INLINE int s_maxi(int x, int y);
SYL_INLINE int s_mini(int x, int y);
SYL_INLINE float s_minf(float x, float y);
SYL_INLINE float s_mod(float x, float y);
SYL_INLINE float s_pow(float Value, float Times);
SYL_INLINE float s_truncatef(float Value, float Remain);
SYL_INLINE double s_truncated(double Value, double Places);
SYL_INLINE float s_normalize(float Value, float Min, float Max);
SYL_INLINE float s_map(float Value, float SourceMin, float SourceMax, float DestMin, float DestMax);
SYL_INLINE svec3 s_rgb_to_hsv(svec3 RGB);
SYL_INLINE svec2 SVEC2F(float a, float b);
SYL_INLINE svec2 SVEC2A(float* a);
SYL_INLINE void s_vec2_zero(svec2* Vector);
SYL_INLINE bool s_vec2_equal(svec2 vec1, svec2 Vec2);
SYL_INLINE bool s_vec2_equal_scalar(svec2 vec1, float Value);
SYL_INLINE bool s_vec2_not_equal(svec2 vec1, svec2 Vec2);
SYL_INLINE bool s_vec2_not_equal_scalar(svec2 vec1, float Value);
SYL_INLINE bool s_vec2_greater(svec2 vec1, svec2 Vec2);
SYL_INLINE bool s_vec2_greater_scalar(svec2 vec1, float Value);
SYL_INLINE bool s_vec2_greater_equal(svec2 vec1, svec2 Vec2);
SYL_INLINE bool s_vec2_greater_equal_scalar(svec2 vec1, float Value);
SYL_INLINE bool s_vec2_less(svec2 vec1, svec2 Vec2);
SYL_INLINE bool s_vec2_less_scalar(svec2 vec1, float Value);
SYL_INLINE bool s_vec2_less_equal(svec2 vec1, svec2 Vec2);
SYL_INLINE bool s_vec2_less_equal_scalar(svec2 vec1, float Value);
SYL_INLINE svec2 s_vec2_add(svec2 vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2p_add(svec2* vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2_add_scalar(svec2 vec1, float Value);
SYL_INLINE svec2 s_vec2p_add_scalar(svec2* vec1, float Value);
SYL_INLINE svec2 s_vec2_sub(svec2 vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2p_sub(svec2* vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2_sub_scalar(svec2 vec1, float Value);
SYL_INLINE svec2 s_scalar_sub_vec2(float Value, svec2 vec1);
SYL_INLINE svec2 s_vec2p_sub_scalar(svec2* vec1, float Value);
SYL_INLINE svec2 s_scalar_sub_vec2p(float Value, svec2* vec1);
SYL_INLINE svec2 s_vec2_mul(svec2 vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2p_mul(svec2* vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2_mul_scalar(svec2 vec1, float Value);
SYL_INLINE svec2 s_vec2p_mul_scalar(svec2* vec1, float Value);
SYL_INLINE svec2 s_vec2_div(svec2 vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2p_div(svec2* vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2_div_scalar(svec2 vec1, float Value);
SYL_INLINE svec2 s_scalar_div_vec2(float Value, svec2 vec1);
SYL_INLINE svec2 s_vec2p_div_scalar(svec2* vec1, float Value);
SYL_INLINE svec2 s_scalar_div_vec2p(float Value, svec2* vec1);
SYL_INLINE svec2 s_vec2_negate(svec2 a);
SYL_INLINE svec2 s_vec2_floor(svec2 A);
SYL_INLINE svec2 s_vec2_round(svec2 A);
SYL_INLINE float s_vec2_dot(svec2 vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2_hadamard(svec2 vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2_lerp(svec2 vec1, svec2 Vec2, float t);
SYL_INLINE svec2 s_vec2_clamp(svec2 Value, svec2 Min, svec2 Max);
SYL_INLINE float s_vec2_length(svec2 vec1);
SYL_INLINE float s_vec2_distance(svec2 vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2_normalize(svec2 a);
SYL_INLINE svec2 s_vec2_reflect(svec2 Pos, svec2 N);
SYL_INLINE svec2 s_vec2_project(svec2 VectorToProject, svec2 ProjectionVector);
SYL_INLINE svec2 s_vec2_max_vector(svec2 vec1, svec2 Vec2);
SYL_INLINE svec2 s_vec2_min_vector(svec2 vec1, svec2 Vec2);
SYL_INLINE float s_vec2_max(svec2 A);
SYL_INLINE float s_vec2_min(svec2 A);
SYL_INLINE float s_vec2_sum(svec2 vec1);
SYL_INLINE float s_triangle_area(svec2 vec1, svec2 Vec2, svec2 Vec3);
SYL_INLINE svec3 SVEC3(float a, float b, float c);
SYL_INLINE svec3 SVEC3A(float* a);
SYL_INLINE void s_vec3_zero(svec3* Vector);
SYL_INLINE bool s_vec3_equal(svec3 vec1, svec3 Vec2);
SYL_INLINE bool s_vec3_equal_scalar(svec3 vec1, float Value);
SYL_INLINE bool s_vec3_not_equal(svec3 vec1, svec3 Vec2);
SYL_INLINE bool s_vec3_not_equal_scalar(svec3 vec1, float Value);
SYL_INLINE bool s_vec3_greater(svec3 vec1, svec3 Vec2);
SYL_INLINE bool s_vec3_less(svec3 vec1, svec3 Vec2);
SYL_INLINE bool s_vec3_less_scalar(svec3 vec1, float Value);
SYL_INLINE bool s_vec3_greater_equal(svec3 vec1, svec3 Vec2);
SYL_INLINE bool s_vec3_greater_equal_scalar(svec3 vec1, float Value);
SYL_INLINE bool s_vec3_less_equal(svec3 vec1, svec3 Vec2);
SYL_INLINE bool s_vec3_less_equal_scalar(svec3 vec1, float Value);
SYL_INLINE svec3 s_vec3_add(svec3 vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3p_add(svec3* vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3_add_scalar(svec3 vec1, float Value);
SYL_INLINE svec3 s_vec3p_add_scalar(svec3* vec1, float Value);
SYL_INLINE svec3 s_vec3_sub(svec3 vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3p_sub(svec3* vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3_sub_scalar(svec3 vec1, float Value);
SYL_INLINE svec3 s_vec3p_sub_saclar(svec3* vec1, float Value);
SYL_INLINE svec3 s_scalar_sub_vec3(float value, svec3 vec1);
SYL_INLINE svec3 s_scalar_sub_vec3p(float value, svec3* vec1);
SYL_INLINE svec3 s_vec3_mul(svec3 vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3p_mul(svec3* vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3_mul_scalar(svec3 vec1, float Value);
SYL_INLINE svec3 s_vec3p_mul_scalar(svec3* vec1, float Value);
SYL_INLINE svec3 s_vec3_div(svec3 vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3p_div(svec3* vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3_div_scalar(svec3 vec1, float Value);
SYL_INLINE svec3 s_vec3p_div_scalar(svec3* vec1, float Value);
SYL_INLINE svec3 s_scalar_div_vec3(float Value, svec3 vec1);
SYL_INLINE svec3 s_scalar_div_vec3p(float Value, svec3* vec1);
SYL_INLINE svec3 s_vec3_floor(svec3 A);
SYL_INLINE svec3 s_vec3_round(svec3 A);
SYL_INLINE svec3 s_vec3_negate(svec3 a);
SYL_INLINE float s_vec3_dot(svec3 vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3_hadamard(svec3 vec1, svec3 Vec2);
SYL_INLINE float s_vec3_length(svec3 vec1);
SYL_INLINE float s_vec3_distance(svec3 vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3_normalize(svec3 a);
SYL_INLINE float s_vec3_max(svec3 A);
SYL_INLINE float s_vec3_min_value(svec3 A);
SYL_INLINE svec3 s_vec3_max_vector(svec3 vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3_min_vector(svec3 vec1, svec3 Vec2);
SYL_INLINE svec3 s_vec3_clamp(svec3 Value, svec3 Min, svec3 Max);
SYL_INLINE svec3 s_vec3_lerp(svec3 vec1, svec3 Vec2, float t);
SYL_INLINE svec3 s_vec3_project(svec3 VectorToProject, svec3 ProjectionVector);
SYL_INLINE svec3 s_vec3_cross(svec3 vec1, svec3 Vec2);
SYL_INLINE float Slope(svec3 PointA, svec3 PointB);
SYL_INLINE svec4 SVEC4(float a, float b, float c, float d);
SYL_INLINE svec4 SVEC4A(float* a);
SYL_INLINE svec4 SVEC4VF(svec3 Vector, float Value);
SYL_INLINE void s_vector4_zero(svec4* Vector);
SYL_INLINE bool s_vec4_equal(svec4 vec1, svec4 Vec2);
SYL_INLINE bool s_vec4_equal_scalar(svec4 vec1, float Value);
SYL_INLINE bool s_vec4_not_equal(svec4 vec1, svec4 Vec2);
SYL_INLINE bool s_vec4_not_equal_scalar(svec4 vec1, float Value);
SYL_INLINE bool s_vec4_greater(svec4 vec1, svec4 Vec2);
SYL_INLINE bool s_vec4_less(svec4 vec1, svec4 Vec2);
SYL_INLINE bool s_vec4_less_scalar(svec4 vec1, float Value);
SYL_INLINE bool s_vec4_greater_equal(svec4 vec1, svec4 Vec2);
SYL_INLINE bool s_vec4_greater_equal_scalar(svec4 vec1, float Value);
SYL_INLINE bool s_vec4_less_equal(svec4 vec1, svec4 Vec2);
SYL_INLINE bool s_vec4_less_equal_scalar(svec4 vec1, float Value);
SYL_INLINE svec4 s_vec4_add(svec4 vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4p_add(svec4* vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4_add_scalar(svec4 vec1, float Value);
SYL_INLINE svec4 s_vec4p_add_scalar(svec4* vec1, float Value);
SYL_INLINE svec4 s_vec4_sub(svec4 vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4p_sub(svec4* vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4_sub_scalar(svec4 vec1, float Value);
SYL_INLINE svec4 s_vec4p_sub_scalar(svec4* vec1, float Value);
SYL_INLINE svec4 s_scalar_vec4_sub(float value, svec4 vec1);
SYL_INLINE svec4 s_scalar_sub_vec4p(float value, svec4* vec1);
SYL_INLINE svec4 s_vec4_mul(svec4 vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4p_mul(svec4* vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4_mul_scalar(svec4 vec1, float Value);
SYL_INLINE svec4 s_vec4p_mul_scalar(svec4* vec1, float Value);
SYL_INLINE svec4 s_vec4_div(svec4 vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4p_div(svec4* vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4_div_scalar(svec4 vec1, float Value);
SYL_INLINE svec4 s_vec4p_div_scalar(svec4* vec1, float Value);
SYL_INLINE svec4 s_vec4_floor(svec4 A);
SYL_INLINE svec4 s_vec4_round(svec4 A);
SYL_INLINE svec4 s_vec4_negate(svec4 a);
SYL_INLINE float s_vec4_dot(svec4 vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4_hadamard(svec4 vec1, svec4 Vec2);
SYL_INLINE float s_vec4_length(svec4 vec1);
SYL_INLINE float s_vec4_distance(svec4 vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4_normalize(svec4 a);
SYL_INLINE svec4 s_vec4_lerp(svec4 vec1, svec4 Vec2, float t);
SYL_INLINE svec4 s_vec4_cross(svec4 vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4_project(svec4 VectorToProject, svec4 ProjectionVector);
SYL_INLINE svec4 s_vec4_max_vector(svec4 vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4_min_vector(svec4 vec1, svec4 Vec2);
SYL_INLINE svec4 s_vec4_clamp(svec4 Value, svec4 Min, svec4 Max);
SYL_INLINE float s_vec4_max(svec4 A);
SYL_INLINE float s_vec4_min(svec4 A);
SYL_INLINE float s_vec4_sum(svec4 vec1);
SYL_INLINE smat4 SMAT4(float value);
SYL_INLINE smat4 SMAT4V(float m00, float m01, float m02, float m03,
		       float m10, float m11, float m12, float m13,
		       float m20, float m21, float m22, float m23,
		       float m30, float m31, float m32, float m33);
SYL_INLINE smat4 SMAT4A(float* a);
SYL_INLINE void s_mat4_zero(smat4* Matrix);
SYL_INLINE smat4 s_mat4_identity();
SYL_INLINE void s_mat4_identityp(smat4 *ptr);
SYL_INLINE bool s_mat4_is_identity(smat4 Mat);
SYL_INLINE smat4 s_mat4_mul(smat4 Matrix1, smat4 Matrix2);
SYL_INLINE smat4 s_mat4_transpose(smat4 Mat);
SYL_INLINE smat4 s_mat4_inverse_noscale(smat4 Matrix);
SYL_INLINE svec4 s_mat4_transform(smat4 Matrix, svec4 Vector);
SYL_INLINE svec4 s_mat4_mul_vec4(smat4 Matrix1, svec4 Vector);
SYL_INLINE svec3 s_mat4_mul_vec3(smat4 Matrix1, svec3 Vector);
SYL_INLINE smat4 s_mat4_translate(smat4 matrix, svec3 vec);
SYL_INLINE smat4 s_mat4_xrotation(float Angle);
SYL_INLINE smat4 s_mat4_yrotation(float Angle);
SYL_INLINE smat4 s_mat4_zrotation(float Angle);
SYL_INLINE smat4 s_mat4_translation(svec3 Vector);
SYL_INLINE smat4 s_mat4_perspective_projection_rh(float Fov, float AspectRatio, float NearClipPlane, float FarClipPlane);
SYL_INLINE smat4 s_mat4_orthographic_projection_rh(float AspectRatio, float NearClipPlane, float FarClipPlane);

#endif // SYLVESTER_H

#ifdef SYL_IMPLEMENTATION

#define SYL_PI 3.14159265359f

#define _SYL_SHUFFLE(a,b,c,d) (((a) << 6) | ((b) << 4) |	\
			       ((c) << 2) | ((d)))

#define _SYL_PERMUTE_PS( v, c ) _mm_shuffle_ps((v), (v), c )
#define _SYL_ADD_PS( a, b, c ) _mm_sub_ps((c), _mm_mul_ps((a), (b)))

#define _SYL_LOAD(a) _mm_load_ps((a))
#define _SYL_LOADV2(a) _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & (a))

#define _SYL_MAKE_SHUFFLE_MASK(x,y,z,w)           (x | (y<<2) | (z<<4) | (w<<6))
#define _SYL_VEC_SWIZZLE_MASK(vec, mask)          _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(vec), mask))
#define _SYL_VEC_SWIZZLE(vec, x, y, z, w)        _SYL_VEC_SWIZZLE_MASK(vec, _SYL_MAKE_SHUFFLE_MASK(x,y,z,w))
#define _SYL_VEC_SWIZZLE1(vec, x)                _SYL_VEC_SWIZZLE_MASK(vec, _SYL_MAKE_SHUFFLE_MASK(x,x,x,x))
#define _SYL_VEC_SWIZZLE_0022(vec)               _mm_moveldup_ps(vec)
#define _SYL_VEC_SWIZZLE_1133(vec)               _mm_movehdup_ps(vec)

// return (vec1[x], vec1[y], vec2[z], vec2[w])
#define _SYL_VEC_SHUFFLE(vec1, vec2, x,y,z,w)    _mm_shuffle_ps(vec1, vec2, _SYL_MAKE_SHUFFLE_MASK(x,y,z,w))
// special shuffle
#define _SYL_VEC_SHUFFLE_0101(vec1, vec2)        _mm_movelh_ps(vec1, vec2)
#define _SYL_VEC_SHUFFLE_2323(vec1, vec2)        _mm_movehl_ps(vec2, vec1)
#define _SYL_SMALL_NUMBER		(1.e-8f)

#ifdef SYL_GENERIC_FUNCTIONS

#define syl_add(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_add,		\
					    float: s_vec2_add_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_add,		\
					     float: s_vec3_add_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_add,		\
					     float: s_vec4_add_scalar	\
					     )				\
)(v1, v2)

#define syl_sub(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_sub,		\
					    float: s_vec2_sub_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_sub,		\
					     float: s_vec3_sub_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_sub,		\
					     float: s_vec4_sub_scalar	\
					    )				\
			     )(v1, v2)


#define syl_mul(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_mul,		\
					    float: s_vec2_mul_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_mul,		\
					     float: s_vec3_mul_scalar,	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_mul,		\
					     float: s_vec4_mul_scalar	\
					     ),				\
			     smat4: _Generic((v2),			\
					      default: s_mat4_mul, \
					      svec4: s_mat4_mul_vec4,	\
					      svec3: s_mat4_mul_vec3 \
					) \
			     )(v1, v2)

#define syl_div(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_div,		\
					    float: s_vec2_div_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_div,		\
					     float: s_vec3_div_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_div,		\
					     float: s_vec4_div_scalar	\
					    )				\
			     )(v1, v2)

#define syl_equal(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_equal,		\
					    float: s_vec2_equal_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_equal,		\
					     float: s_vec3_equal_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_equal,		\
					     float: s_vec4_equal_scalar	\
					     )				\
)(v1, v2)

#define syl_not_equal(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_not_equal,		\
					    float: s_vec2_not_equal_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_not_equal,		\
					     float: s_vec3_not_equal_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_not_equal,		\
					     float: s_vec4_not_equal_scalar	\
					     )				\
)(v1, v2)

#define syl_greater(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_greater,		\
					    float: s_vec2_greater_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_greater,		\
					     float: s_vec3_greater_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_greater,		\
					     float: s_vec4_greater_scalar	\
					     )				\
)(v1, v2)


#define syl_greater_scalar(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_greater_scalar,		\
					    float: s_vec2_greater_scalar_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_greater_scalar,		\
					     float: s_vec3_greater_scalar_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_greater_scalar,		\
					     float: s_vec4_greater_scalar_scalar	\
					     )				\
)(v1, v2)

#define syl_greater_equal(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_greater_equal,		\
					    float: s_vec2_greater_equal_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_greater_equal,		\
					     float: s_vec3_greater_equal_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_greater_equal,		\
					     float: s_vec4_greater_equal_scalar	\
					     )				\
)(v1, v2)

#define syl_greater_equal_scalar(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_greater_equal_scalar,		\
					    float: s_vec2_greater_equal_scalar_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_greater_equal_scalar,		\
					     float: s_vec3_greater_equal_scalar_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_greater_equal_scalar,		\
					     float: s_vec4_greater_equal_scalar_scalar	\
					     )				\
)(v1, v2)

#define syl_less(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_less,		\
					    float: s_vec2_less_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_less,		\
					     float: s_vec3_less_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_less,		\
					     float: s_vec4_less_scalar	\
					     )				\
)(v1, v2)


#define syl_less_scalar(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_less_scalar,		\
					    float: s_vec2_less_scalar_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_less_scalar,		\
					     float: s_vec3_less_scalar_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_less_scalar,		\
					     float: s_vec4_less_scalar_scalar	\
					     )				\
)(v1, v2)

#define syl_less_equal(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_less_equal,		\
					    float: s_vec2_less_equal_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_less_equal,		\
					     float: s_vec3_less_equal_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_less_equal,		\
					     float: s_vec4_less_equal_scalar	\
					     )				\
)(v1, v2)

#define syl_less_equal_scalar(v1, v2) _Generic((v1),					\
			     svec2: _Generic((v2),			\
					    default: s_vec2_less_equal_scalar,		\
					    float: s_vec2_less_equal_scalar_scalar	\
					    ),				\
			     svec3: _Generic((v2),			\
					    default: s_vec3_less_equal_scalar,		\
					     float: s_vec3_less_equal_scalar_scalar	\
					     ),				\
			     svec4: _Generic((v2),			\
					    default: s_vec4_less_equal_scalar,		\
					     float: s_vec4_less_equal_scalar_scalar	\
					     )				\
)(v1, v2)



#define syl_length(vec) _Generic((vec), svec2: s_vec2_length, svec3: s_vec3_length)(vec)
#define syl_lerp(v1, v2, v3) _Generic((v1), float: s_lerp, svec2: s_vec2_lerp, svec3: s_vec3_length, svec4: s_vec4_lerp)(v1, v2, v3)

#define syl_clamp(v1, v2, v3) _Generic((v1), float: s_clampf, double: s_clampd, int: s_clampi, svec2: s_vec2_clamp, svec3: s_vec3_clamp, svec4: s_vec4_clamp)(v1, v2, v3)

#define syl_max2(v1, v2) _Generic((v1), float: s_maxf, int: s_maxi, svec2: s_vec2_max_vector, svec3: s_vec3_max_vector, svec4: s_vec4_max_vector)(v1, v2)

#define syl_max_value(v1) _Generic((v1), svec2: s_vec2_max, svec3: s_vec3_max, svec4: s_vec4_max)(v1)

#define syl_min2(v1, v2) _Generic((v1), float: s_minf, int: s_mini, svec2: s_vec2_min_vector, svec3: s_vec3_min_vector, svec4: s_vec4_min_vector)(v1, v2)

#define syl_min_value(v1) _Generic((v1), svec2: s_vec2_min, svec3: s_vec3_min, svec4: s_vec4_min)(v1)

#define syl_normalize(v1) _Generic((v1), float: s_normalize, svec2: s_vec2_normalize, svec3: s_vec3_normalize, svec4: s_vec4_normalize)(v1)

#define syl_hadamard(v1, v2) _Generic((v1), svec2: s_vec2_hadamard, svec3: s_vec3_hadamard, svec4: s_vec4_hadamard)(v1, v2)

#define syl_dot(v1, v2) _Generic((v1), svec2: s_vec2_dot, svec3: s_vec3_dot, svec4: s_vec4_dot)(v1, v2)
// floor is already reserved for c function
#define syl_floor(v1) _Generic((v1), float: s_floorf, double: s_floord, svec2: s_vec2_floor, svec3: s_vec3_floor, svec4: s_vec4_floor)(v1)

#define syl_round(v1) _Generic((v1), float: s_roundf, double: s_roundd, svec2: s_vec2_round, svec3: s_vec3_round, svec4: s_vec4_round)(v1)

// ceil is already reserved for c function
#define syl_ceil(v1) _Generic((v1), float: s_ceilf, double: s_ceild)(v1)

#define syl_project(v1, v2, v3) _Generic((v1), svec2: s_vec2_project, svec3: s_vec3_project, svec4: s_vec4_project)(v1, v2, v3)

#define syl_sum(v1) _Generic((v1), svec2: s_vec2_sum, svec3: s_vec3_sum, svec4: s_vec4_sum)(v1)
#define syl_negate(v1) _Generic((v1), svec2: s_vec2_negate, svec3: s_vec3_negate, svec4: s_vec4_negate)(v1)

#ifdef SYL_GENERIC_FUNCTIONS_NO_PREFIX_ALIASES
#define add syl_add
#define sub syl_sub

#define mul syl_mul
#define div syl_div

#define equal syl_equal
#define not_equal syl_not_equal

#define greater syl_greater
#define greater_scalar syl_greater_scalar
#define greater_equal syl_greater_equal
#define greater_equal_scalar syl_greater_equal_scalar

#define less syl_less
#define less_scalar syl_less_scalar
#define less_equal syl_less_equal
#define less_equal_scalar syl_less_equal_scalar

#define length syl_length
#define lerp syl_lerp

#define clamp syl_clamp

#define max2 syl_max2
#define max_value syl_max_value

#define min2 syl_min2
#define min_value syl_min_value

#define normalize syl_normalize

#define hadamard syl_hadamard
#define dot syl_dot

#define round syl_round

#define project syl_project
#define sum syl_sum
#define negate syl_negate

#endif

#endif

smat4 _S_IDENT4X4 = { {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f
	} };

svec2 _SVEC2_ZERO = { { 0.0f, 0.0f } };

#if defined(SYL_ENABLE_SSE4)
__m128 _S_XMM_ZERO = { 0.0f, 0.0f, 0.0f, 0.0f };
__m128 _S_IDENT4x4R0 = { 1.0f, 0.0f, 0.0f, 0.0f };
__m128 _S_IDENT4x4R1 = { 0.0f, 1.0f, 0.0f, 0.0f };
__m128 _S_IDENT4x4R2 = { 0.0f, 0.0f, 1.0f, 0.0f };
__m128 _S_IDENT4x4R3 = { 0.0f, 0.0f, 0.0f, 1.0f };
__m128 _S_XMM_MASK_3 = { (float)0xFFFFFFFF, (float)0xFFFFFFFF, (float)0xFFFFFFFF, (float)0x00000000 };
__m128 _S_XMM_MASK_Y = { 0x00000000, (float)0xFFFFFFFF, 0x00000000, 0x00000000 };
#endif

#if defined(SYL_ENABLE_AVX)
__m256 _S_YMM_ZERO = { { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } };
#endif

/*********************************************
 *                 Utility                   *
 *********************************************/


SYL_INLINE float s_radian_to_degree(float Radian)
{
	return(Radian * (180 / SYL_PI));
}

SYL_INLINE float s_degree_to_radian(float Degree)
{
	return (Degree * (SYL_PI / 180));
}

SYL_INLINE float s_roundf(float A)
{
	return((int)(A + 0.5f));
}

SYL_INLINE double s_roundd(double A)
{
	return((int)(A + 0.5f));
}

SYL_INLINE float s_ceilf(float A)
{
	return((int)(A + 1.0f));
}

SYL_INLINE double s_ceild(double A)
{
	return((int)(A + 1.0f));
}

SYL_INLINE float s_floorf(float A)
{
	return((int)A);
}

SYL_INLINE double s_floord(double A)
{
	return((int)A);
}

/* Unpack four 8-bit BGRA values into vec4 */
SYL_INLINE svec4 s_bgra_unpack(int Color)
{
	svec4 Result = { {
			(float)((Color >> 16) & 0xFF),
			(float)((Color >> 8) & 0xFF),
			(float)((Color >> 0) & 0xFF),
			(float)((Color >> 24) & 0xFF)
		} };
	return(Result);
}

/* Pack four 8-bit RGB values */
SYL_INLINE unsigned int s_bgra_pack(svec4 Color)
{
	unsigned int Result =
		((unsigned int)(Color.a) << 24) |
		((unsigned int)(Color.r) << 16) |
		((unsigned int)(Color.g) << 8) |
		((unsigned int)(Color.b) << 0);
	return(Result);
}

/* Unpack four 8-bit RGBA values into vec4 */
SYL_INLINE svec4 s_rgba_unpack(unsigned int Color)
{
	svec4 Result = { {
			(float)((Color >> 24) & 0xFF),
			(float)((Color >> 16) & 0xFF),
			(float)((Color >> 8) & 0xFF),
			(float)((Color >> 0) & 0xFF)
		} };
	return(Result);
}

/* Pack four 8-bit RGBA values */
SYL_INLINE unsigned int s_rgba_pack(svec4 Color)
{
	unsigned int Result =
		((unsigned int)(Color.a) << 24) |
		((unsigned int)(Color.b) << 16) |
		((unsigned int)(Color.g) << 8) |
		((unsigned int)(Color.r) << 0);

	return(Result);

}

SYL_INLINE float s_clampf(float Value, float Min, float Max)
{
	float Result = Value;

	if (Result < Min)
		Result = Min;
	else if (Result > Max)
		Result = Max;

	return(Result);
}

SYL_INLINE float s_clampd(float Value, float Min, float Max)
{
	float Result = Value;

	if (Result < Min)
		Result = Min;
	else if (Result > Max)
		Result = Max;

	return(Result);
}

SYL_INLINE int s_clampi(int Value, int Min, int Max)
{
	int Result = Value;

	if (Result < Min)
		Result = Min;
	else if (Result > Max)
		Result = Max;

	return(Result);
}

SYL_INLINE float s_clamp01f(float Value)
{
	return(s_clampf(0.0f, Value, 1.0f));
}

SYL_INLINE double s_clamp01d(double Value)
{
	return(s_clampf(0.0f, Value, 1.0f));
}

SYL_INLINE float s_clamp_above_zero(float Value)
{
	return (Value < 0) ? 0.0f : Value;
}

SYL_INLINE float s_clamp_below_zero(float Value)
{
	return (Value > 0) ? 0.0f : Value;
}

SYL_INLINE bool s_is_in_range(float Value, float Min, float Max)
{
	return(((Min <= Value) && (Value <= Max)));
}

SYL_INLINE float s_lerp(float A, float t, float B)
{
	return (1.0f - t) * A + t * B;
}

/* It is handy to type square if the expression happens to be very long */
SYL_INLINE float s_square(float x)
{
	return(x * x);
}

/* Absoule value */
SYL_INLINE float s_abs(float x)
{
	return *((unsigned int*)(&x)) &= 0xffffffff >> 1;
}

/* Find the hypotenuse of a triangle given two other sides */
SYL_INLINE float s_pythagorean(float x, float y)
{
	return sqrt(x * x + y * y);
}

/* Maximum of two values */
SYL_INLINE float s_maxf(float x, float y)
{
	if (x > y)
		return(x);
	return(y);
}

SYL_INLINE int s_maxi(int x, int y)
{
	if (x > y)
		return(x);
	return(y);
}

SYL_INLINE int s_mini(int x, int y)
{
	if (x < y)
		return(x);
	return(y);
}

SYL_INLINE float s_minf(float x, float y)
{
	if (x < y)
		return(x);
	return(y);
}

SYL_INLINE float s_mod(float x, float y)
{
	return(x - (s_roundf(x / y) * y));
}

SYL_INLINE float s_pow(float Value, float Times)
{
	float pow = 1;
	for (int i = 0; i < Times; i++) {
		pow = pow * Value;
	}
	return(pow);
}

/* Places -> How many digits you want to keep remained */
SYL_INLINE float s_truncatef(float Value, float Remain)
{
	int Remove = s_pow(10, Remain);
	return(s_roundf(Value * Remove) / Remove);
}

SYL_INLINE double s_truncated(double Value, double Places)
{
	int Remove = pow(10, Places);
	return(s_roundf(Value * Remove) / Remove);
}

SYL_INLINE float s_normalize(float Value, float Min, float Max)
{
	return (Value - Min) / (Max - Min);
}

SYL_INLINE float s_map(float Value, float SourceMin, float SourceMax, float DestMin, float DestMax)
{
	return s_lerp(s_normalize(Value, SourceMin, SourceMax), DestMin, DestMax);
}

SYL_INLINE svec3 s_rgb_to_hsv(svec3 RGB)
{
	/* Range the values between 1 and 0*/
	RGB.r = RGB.r / 255.0;
	RGB.g = RGB.g / 255.0;
	RGB.b = RGB.b / 255.0;

	float MaxValue = s_maxf(RGB.r, s_maxf(RGB.g, RGB.b));
	float MinValue = s_maxf(RGB.r, s_minf(RGB.g, RGB.b));
	float Dif = MaxValue - MinValue;

	svec3 Result;
	Result.x = -1, Result.y = -1;

	if (MaxValue == MinValue)
		Result.x = 0;

	else if (MaxValue == RGB.r)
		Result.x = s_mod(60 * ((RGB.g - RGB.b) / Dif) + 360, 360);

	else if (MaxValue == RGB.g)
		Result.x = s_mod(60 * ((RGB.b - RGB.r) / Dif) + 120, 360);

	else if (MaxValue == RGB.b)
		Result.x = s_mod(60 * ((RGB.r - RGB.g) / Dif) + 240, 360);

	if (MaxValue == 0)
		Result.y = 0;
	else
		Result.y = (Dif / MaxValue) * 100;

	Result.z = MaxValue * 100;
	return(Result);
}

/*********************************************
 *                 VECTOR 2D		  *
 *********************************************/

SYL_INLINE svec2 SVEC2(float a, float b) // From individiaul valeus
{
	svec2 r = { { a, b } };
	return(r);
}

SYL_INLINE svec2 S_VEC2A(float* a) // From array
{
	svec2 r = { { a[0], a[1] } };
	return(r);
}

SYL_INLINE void s_vec2_zero(svec2* vector)
{
	vector->x = 0;
	vector->y = 0;
}

SYL_INLINE bool s_vec2_equal(svec2 vec1, svec2 vec2)
{
	bool result = false;
	if (vec1.x == vec2.x && vec1.y == vec2.y)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_equal_scalar(svec2 vec1, float Value)
{
	bool result = false;
	if (vec1.x == Value && vec1.y == Value)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_not_equal(svec2 vec1, svec2 vec2)
{
	bool result = false;
	if (vec1.x != vec2.x && vec1.y != vec2.y)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_not_equal_scalar(svec2 vec1, float Value)
{
	bool result = false;
	if (vec1.x != Value && vec1.y != Value)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_greater(svec2 vec1, svec2 vec2)
{
	bool result = false;
	if (vec1.x > vec2.x && vec1.y > vec2.y)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_greater_scalar(svec2 vec1, float value)
{
	bool result = false;
	if (vec1.x > value && vec1.y > value)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_greater_equal(svec2 vec1, svec2 vec2)
{
	bool result = false;
	if (vec1.x >= vec2.x && vec1.y >= vec2.y)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_greater_equal_scalar(svec2 vec1, float value)
{
	bool result = false;
	if (vec1.x >= value && vec1.y >= value)
		result = true;
	return(result);
}


SYL_INLINE bool s_vec2_less(svec2 vec1, svec2 vec2)
{
	bool result = false;
	if (vec1.x < vec2.x && vec1.y < vec2.y)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_less_scalar(svec2 vec1, float value)
{
	bool result = false;
	if (vec1.x < value && vec1.y < value)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_less_equal(svec2 vec1, svec2 vec2)
{
	bool result = false;
	if (vec1.x <= vec2.x && vec1.y <= vec2.y)
		result = true;
	return(result);
}

SYL_INLINE bool s_vec2_less_equal_scalar(svec2 vec1, float value)
{
	bool result = false;
	if (vec1.x <= value && vec1.y <= value)
		result = true;
	return(result);
}

SYL_INLINE svec2 s_vec2_add(svec2 vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2));
	return *(svec2*)&r;
#else
	svec2 result = { { (vec1.x + vec2.x), (vec1.y + vec2.y) } };
	return(result);
#endif
}

SYL_INLINE svec2 s_vec2p_add(svec2* vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x + vec2.x;
	vec1->y = vec1->y + vec2.y;	//
	return(*vec1);
#endif
}

SYL_INLINE svec2 s_vec2_add_scalar(svec2 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1), _mm_set1_ps(value));
	return *(svec2*)&r;
#else
	svec2 Result = { { (vec1.x + value), (vec1.y + value) } };
	return(Result);
#endif
}

SYL_INLINE svec2 s_vec2p_add_scalar(svec2* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e), _mm_set1_ps(value));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x + value;
	vec1->y = vec1->y + value;
	return(*vec1);
#endif
}

SYL_INLINE svec2 s_vec2_sub(svec2 vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2));
	return *(svec2*)&r;
#else
	svec2 Result = { { (vec1.x - vec2.x), (vec2.x - vec2.y) } };
	return(Result);
#endif
}

SYL_INLINE svec2 s_vec2p_sub(svec2* vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x - vec2.x;
	vec1->y = vec1->y - vec2.y;
	return(*vec1);
#endif
}

SYL_INLINE svec2 s_vec2_sub_scalar(svec2 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1), _mm_set1_ps(value));
	return *(svec2*)&r;
#else
	svec2 Result = { { (vec1.x - value), (vec1.y - value) } };
	return(Result);
#endif
}

SYL_INLINE svec2 s_scalar_sub_vec2(float value, svec2 vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_set1_ps(value), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1));
	return *(svec2*)&r;
#else
	svec2 Result = { { (value - vec1.x), (value - vec1.y) } };
	return(Result);
#endif
}

SYL_INLINE svec2 s_vec2p_sub_scalar(svec2* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e), _mm_set1_ps(value));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x - value;
	vec1->y = vec1->y - value;
	return(*vec1);
#endif
}

SYL_INLINE svec2 s_scalar_sub_vec2p(float value, svec2* vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e), _mm_set1_ps(value));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = value - vec1->x;
	vec1->y = value - vec1->y;
	return(*vec1);
#endif
}

SYL_INLINE svec2 s_vec2_mul(svec2 vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2));
	return *(svec2*)&r;
#else
	svec2 Result = { { (vec1.x * vec2.x), (vec1.y * vec2.y) } };
	return(Result);
#endif
}

SYL_INLINE svec2 s_vec2p_mul(svec2* vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x * vec2.x;
	vec1->y = vec1->y * vec2.y;
	return(*vec1);
#endif
}

SYL_INLINE svec2 s_vec2_mul_scalar(svec2 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1), _mm_set1_ps(value));
	return *(svec2*)&r;
#else
	svec2 Result = { { (vec1.x * value), (vec1.y * value) } };
	return(Result);
#endif
}

SYL_INLINE svec2 s_vec2p_mul_scalar(svec2* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e), _mm_set1_ps(value));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x * value;
	vec1->y = vec1->y * value;
	return(*vec1);
#endif
}

SYL_INLINE svec2 s_vec2_div(svec2 vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2));
	return *(svec2*)&r;
#else
	svec2 Result = { { (vec1.x / vec2.x), (vec1.y / vec2.y) } };
	return(Result);
#endif
}

SYL_INLINE svec2 s_vec2p_div(svec2* vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x / vec2.x;
	vec1->y = vec1->y / vec2.y;
	return(*vec1);
#endif
}

SYL_INLINE svec2 s_vec2_div_scalar(svec2 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1), _mm_set1_ps(value));
	return *(svec2*)&r;
#else
	svec2 Result = { { (vec1.x / value), (vec1.y / value) } };
	return(Result);
#endif
}

SYL_INLINE svec2 s_scalar_div_vec2(float value, svec2 vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_set1_ps(value), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1));
	return *(svec2*)&r;
#else
	svec2 Result = { { (value / vec1.x), (value / vec1.y) } };
	return(Result);
#endif
}

SYL_INLINE svec2 s_vec2p_div_scalar(svec2* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e), _mm_set1_ps(value));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x / value;
	vec1->y = vec1->y / value;;
	return(*vec1);
#endif
}

SYL_INLINE svec2 s_scalar_div_vec2p(float value, svec2* vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_set1_ps(value), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1->e));
	*vec1 = *(svec2*)&r;
	return(*vec1);
#else
	vec1->x = value / vec1->x;
	vec1->y = value / vec1->y;
	return(*vec1);
#endif
}

/* Negate all components of the vector */
SYL_INLINE svec2 s_vec2_negate(svec2 a)
{
	svec2 Result = { { -a.x, -a.y } };
	return(Result);
}

SYL_INLINE svec2 s_vec2_floor(svec2 A)
{
	svec2 Result = { { s_floorf(A.x), s_floorf(A.y) } };
	return(Result);
}

/* Round all components of the vec3 to nearest integer*/
SYL_INLINE svec2 s_vec2_round(svec2 A)
{
	svec2 Result = { { s_roundf(A.x), s_roundf(A.y) } };
	return(Result);
}

SYL_INLINE float s_vec2_dot(svec2 vec1, svec2 vec2)
{
	return((vec1.x * vec2.x) + (vec1.y * vec2.y));
}

SYL_INLINE svec2 s_vec2_hadamard(svec2 vec1, svec2 vec2)
{
	svec2 Result = { { (vec1.x * vec2.x), (vec1.y * vec2.y) } };
	return(Result);
}

SYL_INLINE svec2 s_vec2_lerp(svec2 vec1, svec2 vec2, float t)
{
	svec2 r = { { vec1.x + (vec2.x - vec1.x * t), vec1.y + ((vec2.y - vec1.y * t)) } };
	return r;
}

SYL_INLINE svec2 s_vec2_clamp(svec2 value, svec2 Min, svec2 Max)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_min_ps(_mm_max_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & value.v), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Min.v)), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Max.v));
	return *(svec2*)&Result;
#else
	svec2 Result = { { s_clampf(value.x, Min.x, Max.x), s_clampf(value.y, Min.y, Max.y) } };
	return(Result);
#endif
}

SYL_INLINE float s_vec2_length(svec2 vec1)
{
	return(sqrt((vec1.x * vec1.x) + (vec1.y * vec1.y)));
}

SYL_INLINE float s_vec2_distance(svec2 vec1, svec2 vec2)
{
	return s_vec2_length(s_vec2_sub(vec1, vec2));
}

SYL_INLINE svec2 s_vec2_normalize(svec2 a)
{
	return(s_vec2_mul_scalar(a, (1.0f / s_vec2_length(a))));
}

/* Reflect a position to a normal plane */
SYL_INLINE svec2 s_vec2_reflect(svec2 Pos, svec2 N)
{
	svec2 Normal = s_vec2_normalize(N);
	float r = s_vec2_dot(Pos, Normal);
	svec2 t = { { (float)(Pos.e[0] - Normal.e[0] * 2.0 * r), (float)(Pos.e[1] - Normal.e[1] * 2.0 * r) } };
	return(t);
}

/* Project from a position along a vector on to a plane */
SYL_INLINE svec2 s_vec2_project(svec2 vectorToProject, svec2 Projectionvector)
{
	float scale = s_vec2_dot(Projectionvector, vectorToProject) / s_vec2_dot(Projectionvector, Projectionvector);
	return(s_vec2_mul_scalar(Projectionvector, scale));
}

/* Flattens a position to a normal plane */
static inline svec2 s_vec2_flatten(svec2 Pos, svec2 Normal)
{
	float f = s_vec2_dot(Pos, Normal);
	svec2 result = { { (Pos.e[0] - Normal.e[0] * f), (Pos.e[1] - Normal.e[1] * f) } };
	return(result);
}

/* Per component comparsion to return a vector containing the largest components */
SYL_INLINE svec2 s_vec2_max_vector(svec2 vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_max_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1.e), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2.e));
	return *(svec2*)&r;
#else
	svec2 Result;
	if (vec1.x > vec2.x)
		Result.x = vec1.x;
	else
		Result.x = vec2.x;

	if (vec1.y > vec2.y)
		Result.y = vec1.y;
	else
		Result.y = vec2.y;
	return(Result);
#endif
}

/* Per component comparsion to return a vector containing the smollest components */
SYL_INLINE svec2 s_vec2_min_vector(svec2 vec1, svec2 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_min_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1.e), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2.e));
	return *(svec2*)&r;
#else
	svec2 Result;
	if (vec1.x < vec2.x)
		Result.x = vec1.x;
	else
		Result.x = vec2.x;

	if (vec1.y < vec2.y)
		Result.y = vec1.y;
	else
		Result.y = vec2.y;
	return(Result);
#endif
}

/* Return the biggest element inside vec4 */
SYL_INLINE float s_vec2_max(svec2 A)
{
	if (A.e[0] > A.e[1])
		return(A.e[0]);
	else
		return(A.e[1]);
}

/* Return the smollest element inside vec4 */
SYL_INLINE float s_vec2_min(svec2 A)
{
	if (A.e[0] < A.e[1])
		return(A.e[0]);
	else
		return(A.e[1]);
}

/* Add all components of the vector together */
SYL_INLINE float s_vec2_sum(svec2 vec1)
{
	return(vec1.x + vec1.y);
}

SYL_INLINE float s_triangle_area(svec2 vec1, svec2 vec2, svec2 Vec3)
{
	float r = ((vec1.x * vec2.y) + (vec2.x * Vec3.y) + (Vec3.x * vec1.y) - (vec1.y * vec2.x) - (vec2.y * Vec3.x) - (Vec3.y * vec1.x)) / 2;
	if (r < 0)
		{
			r = -r;
		}
	return(r);
}

/*********************************************
 *                 VECTOR 3D		     *
 *********************************************/

SYL_INLINE svec3 SVEC3(float a, float b, float c)
{
	svec3 r = { { a, b, c } };
	return(r);
}

SYL_INLINE svec3 SVEC3A(float* a)
{
	svec3 r = { { a[0], a[1], a[2] } };
	return(r);
}

SYL_INLINE void s_vec3_zero(svec3* vector)
{
	vector->x = 0;
	vector->y = 0;
	vector->z = 0;
}

SYL_INLINE bool s_vec3_equal(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpeq_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x == vec2.x) && (vec1.y == vec2.y) && (vec1.z == vec2.z))
		return(true);
	else
		return(false);
#endif
}

SYL_INLINE bool s_vec3_equal_scalar(svec3 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpeq_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x == value) && (vec1.y == value) && (vec1.z == value))
		return(true);
	else
		return(false);

#endif
}

SYL_INLINE bool s_vec3_not_equal(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)

	__m128 Result = _mm_cmpeq_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return (((_mm_movemask_ps(Result) & 7) == 7) == 0);
#else
	if ((vec1.x != vec2.x) && (vec1.y != vec2.y) && (vec1.z == vec2.z))
		return(true);
	else
		return(false);
#endif
}

SYL_INLINE bool s_vec3_not_equal_scalar(svec3 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpneq_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x != value) && (vec1.y != value) && (vec1.z == value))
		return(true);
	else
		return(false);
#endif
}

SYL_INLINE bool s_vec3_greater(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)

	__m128 Result = _mm_cmpgt_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x > vec2.x) && (vec1.y > vec1.y) && (vec1.z > vec1.z))
		return(true);
	else
		return(false);
#endif
}

SYL_INLINE bool s_vec3_less(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmplt_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x < vec2.x) && (vec1.y < vec2.y) && (vec1.z < vec2.z))
		return(true);
	else
		return(false);
#endif
}

SYL_INLINE bool s_vec3_less_scalar(svec3 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmplt_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x < value) && (vec1.y < value) && (vec1.z < value))
		return(true);
	else
		return(false);
#endif
}


SYL_INLINE bool s_vec3_greater_equal(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpge_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x >= vec2.x) && (vec1.y >= vec2.y) && (vec1.z > vec2.z))
		return(true);
	else
		return(false);
#endif
}

SYL_INLINE bool s_vec3_greater_equal_scalar(svec3 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpge_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x >= value) && (vec1.y >= value) && (vec1.z > value))
		return(true);
	else
		return(false);
#endif
}

SYL_INLINE bool s_vec3_less_equal(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmple_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x <= vec2.x) && (vec1.y <= vec2.y) && (vec1.z > vec2.z))
		return(true);
	else
		return(false);
#endif
}

SYL_INLINE bool s_vec3_less_equal_scalar(svec3 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmple_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
	if ((vec1.x <= value) && (vec1.y <= value) && (vec1.z > value))
		return(true);
	else
		return(false);
#endif
}

SYL_INLINE svec3 s_vec3_add(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & vec2));
	return *(svec3*)&r;
#else
	svec3 Result = { { (vec1.x + vec2.x), (vec1.y + vec2.y), (vec1.z + vec2.z) } };
	return(Result);
#endif
}

SYL_INLINE svec3 s_vec3p_add(svec3* vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(_mm_load_ps(vec1->e), _mm_load_ps(vec2.e));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x + vec2.x;
	vec1->y = vec1->y + vec2.y;
	vec1->z = vec1->z + vec2.z;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_vec3_add_scalar(svec3 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return *(svec3*)&r;
#else
	svec3 Result = { { (vec1.x + value), (vec1.y + value), (vec1.z + value) } };
	return(Result);
#endif
}

SYL_INLINE svec3 s_vec3p_add_scalar(svec3* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(_mm_load_ps(vec1->e), _mm_set1_ps(value));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x + value;
	vec1->y = vec1->y + value;
	vec1->z = vec1->z + value;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_vec3_sub(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return *(svec3*)&r;
#else
	svec3 Result = { { (vec1.x - vec2.x), (vec1.y - vec2.y), (vec1.z - vec2.z) } };
	return(Result);
#endif
}

SYL_INLINE svec3 s_vec3p_sub(svec3* vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_load_ps(vec1->e), _mm_load_ps(vec2.e));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x - vec2.x;
	vec1->y = vec1->y - vec2.y;
	vec1->z = vec1->z - vec2.z;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_vec3_sub_scalar(svec3 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return *(svec3*)&r;
#else
	svec3 Result = { { (vec1.x - value), (vec1.y - value), (vec1.z - value) } };
	return(Result);
#endif
}

SYL_INLINE svec3 s_vec3p_sub_saclar(svec3* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_load_ps(vec1->e), _mm_set1_ps(value));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x - value;
	vec1->y = vec1->y - value;
	vec1->z = vec1->z - value;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_scalar_sub_vec3(float value, svec3 vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_set1_ps(value), _mm_load_ps(vec1.e));
	return *(svec3*)&r;
#else
	svec3 result = { { (value - vec1.x), (value - vec1.y), (value - vec1.z) } };
	return(result);
#endif
}

SYL_INLINE svec3 s_scalar_sub_vec3p(float value, svec3* vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_set1_ps(value), _mm_load_ps(vec1->e));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = value - vec1->x;
	vec1->y = value - vec1->y;
	vec1->z = value - vec1->z;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_vec3_mul(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return *(svec3*)&r;
#else
	svec3 Result = { { (vec1.x * vec2.x), (vec1.y * vec2.y), (vec1.z * vec2.z) } };
	return(Result);
#endif
}

SYL_INLINE svec3 s_vec3p_mul(svec3* vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(_mm_load_ps(vec1->e), _mm_load_ps(vec2.e));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x * vec2.x;
	vec1->y = vec1->y * vec2.y;
	vec1->z = vec1->z * vec2.z;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_vec3_mul_scalar(svec3 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return *(svec3*)&r;
#else
	svec3 Result = { { (vec1.x * value), (vec1.y * value),  (vec1.z * value) } };
	return(Result);
#endif
}

SYL_INLINE svec3 s_vec3p_mul_scalar(svec3* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(_mm_load_ps(vec1->e), _mm_set1_ps(value));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x * value;
	vec1->y = vec1->y * value;
	vec1->z = vec1->z * value;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_vec3_div(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return *(svec3*)&r;
#else
	svec3 Result = { { (vec1.x / vec2.x), (vec1.y / vec2.y), (vec1.z / vec2.z) } };
	return(Result);
#endif
}

SYL_INLINE svec3 s_vec3p_div(svec3* vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_load_ps(vec1->e), _mm_load_ps(vec2.e));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x / vec2.x;
	vec1->y = vec1->y / vec2.y;
	vec1->z = vec1->z / vec2.z;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_vec3_div_scalar(svec3 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return *(svec3*)&r;
#else
	svec3 Result = { { (vec1.x / value), (vec1.y / value), (vec1.z / value) } };
	return(Result);
#endif
}

SYL_INLINE svec3 s_vec3p_div_scalar(svec3* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_load_ps(vec1->e), _mm_set1_ps(value));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x / value;
	vec1->y = vec1->y / value;
	vec1->z = vec1->z / value;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_scalar_div_vec3(float value, svec3 vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_set1_ps(value), _mm_load_ps(vec1.e));
	return *(svec3*)&r;
#else
	svec3 Result = { { (value / vec1.x), (value / vec1.y), (value / vec1.z) } };
	return(Result);
#endif
}

SYL_INLINE svec3 s_scalar_div_vec3p(float value, svec3* vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(_mm_set1_ps(value), _mm_load_ps(vec1->e));
	*vec1 = *(svec3*)&r;
	return(*vec1);
#else
	vec1->x = value / vec1->x;
	vec1->y = value / vec1->y;
	vec1->z = value / vec1->z;
	return(*vec1);
#endif
}

SYL_INLINE svec3 s_vec3_floor(svec3 A)
{
	svec3 Result = { { s_floorf(A.x), s_floorf(A.y), s_floorf(A.z) } };
	return(Result);
}

/* Round all components of the vec3 to nearest integer*/
SYL_INLINE svec3 s_vec3_round(svec3 A)
{
	svec3 Result = { { s_roundf(A.x), s_roundf(A.y), s_roundf(A.z) } };
	return(Result);
}

/* Negate all components of the vector */
SYL_INLINE svec3 s_vec3_negate(svec3 a)
{
	svec3 r = { { -a.x, -a.y, -a.z } };
	return(r);
}

SYL_INLINE float s_vec3_dot(svec3 vec1, svec3 vec2)
{
	return((vec1.x * vec2.x) + (vec1.y * vec2.y) + (vec1.z * vec2.z));
}

SYL_INLINE svec3 s_vec3_hadamard(svec3 vec1, svec3 vec2)
{
	svec3 Result = { { (vec1.x * vec2.x), (vec1.y * vec2.y), (vec1.z * vec1.z) } };
	return(Result);
}

SYL_INLINE float s_vec3_length(svec3 vec1)
{
	return(sqrt((vec1.x * vec1.x) + (vec1.y * vec1.y) + (vec1.z * vec1.z)));
}

SYL_INLINE float s_vec3_distance(svec3 vec1, svec3 vec2)
{
	return s_vec3_length(s_vec3_sub(vec1, vec2));
}

SYL_INLINE svec3 s_vec3_normalize(svec3 a)
{
	return(s_vec3_mul_scalar(a, (1.0f / s_vec3_length(a))));
}

/* Return the biggest element inside vec4 */
SYL_INLINE float s_vec3_max(svec3 A)
{
	/* NOTE(hsnovel): Current SSE4 version doesn't work. It is removed for now. Reimplement it later.*/
	if (A.e[0] >= A.e[1] && A.e[0] >= A.e[2])
		return(A.e[0]);
	if (A.e[1] >= A.e[0] && A.e[1] >= A.e[2])
		return(A.e[1]);
	if (A.e[2] >= A.e[0] && A.e[2] >= A.e[1])
		return(A.e[2]);

	return(A.e[0]); // HMMMMMMMMMMMMMMMMMMMM
}

/* Return the smollest element inside vec4 */
SYL_INLINE float s_vec3_min_value(svec3 A)
{
	/* NOTE(hsnovel): Current SSE4 version doesn't work. It is removed for now. Reimplement it later.*/
	if (A.e[0] <= A.e[1] && A.e[0] <= A.e[2])
		return(A.e[0]);
	if (A.e[1] <= A.e[0] && A.e[1] <= A.e[2])
		return(A.e[1]);
	if (A.e[2] <= A.e[0] && A.e[2] <= A.e[1])
		return(A.e[2]);

	return(A.e[0]); // HMMMMMMMMMMMMMMMMMMMM
}

/* Per component comparsion to return a vector containing the largest components */
SYL_INLINE svec3 s_vec3_max_vector(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_max_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return *(svec3*)&r;
#else
	svec3 Result;
	if (vec1.x > vec2.x)
		Result.x = vec1.x;
	else
		Result.x = vec2.x;

	if (vec1.y > vec2.y)
		Result.y = vec1.y;
	else
		Result.y = vec2.y;

	if (vec1.z > vec2.z)
		Result.z = vec1.z;
	else
		Result.z = vec2.z;

	return(Result);
#endif
}

/* Per component comparsion to return a vector containing the smollest components */
SYL_INLINE svec3 s_vec3_min_vector(svec3 vec1, svec3 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_min_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return *(svec3*)&r;
#else
	svec3 Result;
	if (vec1.x < vec2.x)
		Result.x = vec1.x;
	else
		Result.x = vec2.x;

	if (vec1.y < vec2.y)
		Result.y = vec1.y;
	else
		Result.y = vec2.y;

	if (vec1.z < vec2.z)
		Result.z = vec1.z;
	else
		Result.z = vec2.z;

	return(Result);
#endif
}

SYL_INLINE svec3 s_vec3_clamp(svec3 value, svec3 Min, svec3 Max)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_max_ps(_mm_setr_ps(Min.x, Min.y, Min.z, 0.0f), _mm_setr_ps(value.x, value.y, value.z, 0.0f));
	Result = _mm_min_ps(_mm_setr_ps(Max.x, Max.y, Max.z, 0.0f), Result);
	return *(svec3*)&Result;
#else
	svec3 Result = s_vec3_min_vector(s_vec3_max_vector(value, Min), Max);
	return(Result);
#endif
}

/* TODO(hsnovel): Faster ! */
SYL_INLINE svec3 s_vec3_lerp(svec3 vec1, svec3 vec2, float t)
{
	svec3 Result = { { vec1.e[0] + ((vec2.e[0] - vec1.e[0]) * t),
				   vec1.e[1] + ((vec2.e[1] - vec1.e[1]) * t),
				   vec1.e[2] + ((vec2.e[2] - vec1.e[2]) * t) } };
	return(Result);

}

SYL_INLINE svec3 s_vec3_project(svec3 vectorToProject, svec3 Projectionvector)
{
	float scale = s_vec3_dot(Projectionvector, vectorToProject) / s_vec3_dot(Projectionvector, Projectionvector);
	return(s_vec3_mul_scalar(Projectionvector, scale));
}

SYL_INLINE svec3 s_vec3_cross(svec3 vec1, svec3 vec2)
{
#if defined(SYL_DEBUG)
	svec3 Result;
	Result.x = vec1.y * vec2.z - vec1.z * vec2.y;
	Result.y = vec1.z * vec2.x - vec1.x * vec2.z;
	Result.z = vec1.x * vec2.y - vec1.y * vec2.x;
	return(Result);
#else
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 V1 = _mm_load_ps(vec1.e);
	__m128 V2 = _mm_load_ps(vec2.e);
	__m128 A1 = _SYL_PERMUTE_PS(V1, _SYL_SHUFFLE(3, 0, 2, 1));
	__m128 A2 = _SYL_PERMUTE_PS(V2, _SYL_SHUFFLE(3, 1, 0, 2));
	__m128 r = _mm_mul_ps(A1, A2);
	A1 = _SYL_PERMUTE_PS(A1, _MM_SHUFFLE(3, 0, 2, 1));
	A2 = _SYL_PERMUTE_PS(A2, _MM_SHUFFLE(3, 1, 0, 2));
	r = _SYL_ADD_PS(A1, A2, r);
	__m128 re = _mm_and_ps(r, _S_XMM_MASK_3);
	return *(svec3*)&re;
#else
	svec3 Result = { { vec1.e[1] * vec2.e[2] - vec1.e[2] * vec2.e[1],
				   vec1.e[2] * vec2.e[0] - vec1.e[0] * vec2.e[2],
				   vec1.e[0] * vec2.e[1] - vec1.e[1] * vec2.e[0] } };
	return(Result);
#endif
#endif
}

/* TODO(hsnovel): Typed this out of my mind, might be incorrect
   check the function later. Currently undocumented. */
SYL_INLINE float Slope(svec3 PointA, svec3 PointB)
{
	float Adjacent = sqrt(s_square(PointA.z - PointB.z) + s_square(PointA.x - PointB.x));
	float Opposite = s_abs(PointA.y - PointB.y);
	return(tanf(Opposite / Adjacent));
}

//SYL_INLINE float AreaOfTriangle(vec3 a, vec3 b, vec3 c)
//{
//	vec3 vector = a - b;
//	float Area = sqrt(Dot(vector, vector));
//	vec3 Side = c - b;
//	float f = Dot(vector, Side);
//	Side -= vector * f;
//	Area *= sqrtf(Dot(Side, Side));
//	return(Area);
//}

/*********************************************
 *                   VECTOR 4D		         *
 *********************************************/

SYL_INLINE svec4 SVEC4(float a, float b, float c, float d)
{
	svec4 r = { { a, b, c, d } };
	return(r);
}

SYL_INLINE svec4 SVEC4A(float* a)
{
	svec4 r = { { a[0], a[1], a[2], a[3] } };
	return(r);
}

SYL_INLINE svec4 SVEC4VF(svec3 vector, float value)
{
	svec4 r = { { vector.x, vector.y, vector.z, value } };
	return(r);
}

SYL_INLINE void s_vector4_zero(svec4* vector)
{
#if defined(SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	_mm_store_ps(vector->e, _S_XMM_ZERO);
#else
	vector->x = 0;
	vector->y = 0;
	vector->z = 0;
	vector->w = 0;
#endif
}

SYL_INLINE bool s_vec4_equal(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpeq_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x == vec2.x) && (vec1.y == vec2.y) && (vec1.z == vec2.z)) {
		return(true);
	}
	else {
		return(false);
	}
#endif
}

SYL_INLINE bool s_vec4_equal_scalar(svec4 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpeq_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x == value) && (vec1.y == value) && (vec1.z == value)) {
		return(true);
	}
	else {
		return(false);
	}
#endif
}

SYL_INLINE bool s_vec4_not_equal(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpneq_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x != vec2.x) && (vec1.y != vec2.y) && (vec1.z != vec2.z)) {
		return(true);
	}
	else {
		return(false);
	}

#endif
}

SYL_INLINE bool s_vec4_not_equal_scalar(svec4 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpneq_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x != value) && (vec1.y != value) && (vec1.z != value)) {
		return(true);
	}
	else {
		return(false);
	}
#endif
}

SYL_INLINE bool s_vec4_greater(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpgt_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x > vec2.x) && (vec1.y > vec2.y) && (vec1.z > vec2.y)) {
		return(true);
	}
	else {
		return(false);
	}
#endif
}

SYL_INLINE bool s_vec4_less(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmplt_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x < vec2.x) && (vec1.y < vec2.y) && (vec1.z < vec2.y)) {
		return(true);
	}
	else {
		return(false);
	}
#endif
}

SYL_INLINE bool s_vec4_less_scalar(svec4 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmplt_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x < value) && (vec1.y < value) && (vec1.z < value)) {
		return(true);
	}
	else {
		return(false);
	}
#endif
}


SYL_INLINE bool s_vec4_greater_equal(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpge_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x >= vec2.x) && (vec1.y >= vec2.y) && (vec1.z >= vec2.y)) {
		return(true);
	}
	else {
		return(false);
	}
#endif
}

SYL_INLINE bool s_vec4_greater_equal_scalar(svec4 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmpge_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x >= value) && (vec1.y >= value) && (vec1.z >= value)) {
		return(true);
	}
	else {
		return(false);
	}
#endif
}

SYL_INLINE bool s_vec4_less_equal(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmple_ps(_mm_load_ps(vec1.e), _mm_load_ps(vec2.e));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x <= vec2.x) && (vec1.y <= vec2.y) && (vec1.z <= vec2.y)) {
		return(true);
	}
	else {
		return(false);
	}

#endif
}

SYL_INLINE bool s_vec4_less_equal_scalar(svec4 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_cmple_ps(_mm_load_ps(vec1.e), _mm_set1_ps(value));
	return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
	if ((vec1.x <= value) && (vec1.y <= value) && (vec1.z <= value)) {
		return(true);
	}
	else {
		return(false);
	}
#endif
}

SYL_INLINE svec4 s_vec4_add(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(vec1.v, vec2.v);
	return *(svec4*)&r;
#else
	svec4 Result = { { (vec1.x + vec2.x), (vec1.y + vec2.y), (vec1.z + vec2.z), (vec1.w + vec2.w) } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4p_add(svec4* vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(vec1->v, vec2.v);
	*vec1 = *(svec4*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x + vec2.x;
	vec1->y = vec1->y + vec2.y;
	vec1->z = vec1->z + vec2.z;
	vec1->w = vec1->w + vec2.w;
	return(*vec1);
#endif
}

SYL_INLINE svec4 s_vec4_add_scalar(svec4 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(vec1.v, _mm_set1_ps(value));
	return *(svec4*)&r;
#else
	svec4 Result = { { (vec1.x + value), (vec1.y + value), (vec1.z + value), (vec1.w + value) } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4p_add_scalar(svec4* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_add_ps(vec1->v, _mm_set1_ps(value));
	*vec1 = *(svec4*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x + value;
	vec1->y = vec1->y + value;
	vec1->z = vec1->z + value;
	vec1->w = vec1->w + value;
	return(*vec1);
#endif
}

SYL_INLINE svec4 s_vec4_sub(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(vec1.v, vec2.v);
	return *(svec4*)&r;
#else
	svec4 Result = { { (vec1.x - vec2.x), (vec1.y - vec2.y), (vec1.z - vec2.z), (vec1.w - vec2.w) } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4p_sub(svec4* vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(vec1->v, vec2.v);
	*vec1 = *(svec4*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x - vec2.x;
	vec1->y = vec1->y - vec2.y;
	vec1->z = vec1->z - vec2.z;
	vec1->w = vec1->w - vec2.w;
	return(*vec1);
#endif
}

SYL_INLINE svec4 s_vec4_sub_scalar(svec4 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(vec1.v, _mm_set1_ps(value));
	return *(svec4*)&r;
#else
	svec4 Result = { { (vec1.x - value), (vec1.y - value), (vec1.z - value), (vec1.w - value) } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4p_sub_scalar(svec4* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(vec1->v, _mm_set1_ps(value));
	*vec1 = *(svec4*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x - value;
	vec1->y = vec1->y - value;
	vec1->z = vec1->z - value;
	vec1->w = vec1->w - value;
	return(*vec1);
#endif
}

SYL_INLINE svec4 s_scalar_vec4_sub(float value, svec4 vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_set1_ps(value), vec1.v);
	return *(svec4*)&r;
#else
	svec4 Result = { { (value - vec1.x), (value - vec1.y), (value - vec1.z), (value - vec1.w) } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_scalar_sub_vec4p(float value, svec4* vec1)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_sub_ps(_mm_set1_ps(value), vec1->v);
	*vec1 = *(svec4*)&r;
	return(*vec1);
#else
	vec1->x = value - vec1->x;
	vec1->y = value - vec1->y;
	vec1->z = value - vec1->z;
	vec1->w = value - vec1->w;
	return(*vec1);
#endif
}


SYL_INLINE svec4 s_vec4_mul(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(vec1.v, _mm_load_ps(vec2.e));
	return *(svec4*)&r;
#else
	svec4 Result = { { (vec1.x - vec2.x), (vec1.y - vec2.y), (vec1.z - vec2.z), (vec1.w - vec2.w) } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4p_mul(svec4* vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(vec1->v, _mm_load_ps(vec2.e));
	*vec1 = *(svec4*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x * vec2.x;
	vec1->y = vec1->y * vec2.y;
	vec1->z = vec1->z * vec2.z;
	vec1->w = vec1->w * vec2.w;
	return(*vec1);
#endif
}

SYL_INLINE svec4 s_vec4_mul_scalar(svec4 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(vec1.v, _mm_set1_ps(value));
	return *(svec4*)&r;
#else
	svec4 Result = { { (vec1.x * value), (vec1.y * value), (vec1.z * value), (vec1.w * value) } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4p_mul_scalar(svec4* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_mul_ps(vec1->v, _mm_set1_ps(value));
	*vec1 = *(svec4*)&r;
	return(*vec1);
#else
	vec1->x = vec1->x * value;
	vec1->y = vec1->y * value;
	vec1->z = vec1->z * value;
	vec1->w = vec1->w * value;
	return(*vec1);
#endif
}

SYL_INLINE svec4 s_vec4_div(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 r = _mm_div_ps(vec1.v, vec2.v);
	return *(svec4*)&r;
#else
	svec4 Result = { { (vec1.x / vec2.x), (vec1.y / vec2.y), (vec1.z / vec2.z), (vec1.w / vec2.w) } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4p_div(svec4* vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_div_ps(vec1->v, vec2.v);
	*vec1 = *(svec4*)&Result;
	return(*vec1);
#else
	vec1->x = vec1->x / vec2.x;
	vec1->y = vec1->y / vec2.y;
	vec1->z = vec1->z / vec2.z;
	vec1->w = vec1->w / vec2.w;
	return(*vec1);
#endif
}

SYL_INLINE svec4 s_vec4_div_scalar(svec4 vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_div_ps(vec1.v, _mm_set1_ps(value));
	return *(svec4*)&Result;
#else
	svec4 Result = { { (vec1.x / value), (vec1.y / value), (vec1.z / value), (vec1.w / value) } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4p_div_scalar(svec4* vec1, float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_div_ps(vec1->v, _mm_set1_ps(value));
	*vec1 = *(svec4*)&Result;
	return(*vec1);
#else
	vec1->x = vec1->x / value;
	vec1->y = vec1->y / value;
	vec1->z = vec1->z / value;
	vec1->w = vec1->w / value;
	return(*vec1);
#endif
}

SYL_INLINE svec4 s_vec4_floor(svec4 A)
{
	svec4 Result = { { s_floorf(A.x), s_floorf(A.y), s_floorf(A.z), s_floorf(A.w) } };
	return(Result);
}

/* Round all components of the vec3 to nearest integer*/
SYL_INLINE svec4 s_vec4_round(svec4 A)
{
	svec4 Result = { { s_roundf(A.x), s_roundf(A.y), s_roundf(A.z), s_roundf(A.w) } };
	return(Result);
}

/* Negate all components of the vector */
SYL_INLINE svec4 s_vec4_negate(svec4 a)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_sub_ps(_mm_set1_ps(0), a.v);
	return *(svec4*)&Result;
#else
	svec4 Result = { { -a.x, -a.y, -a.z, -a.w } };
	return(Result);
#endif
}

SYL_INLINE float s_vec4_dot(svec4 vec1, svec4 vec2)
{
#if defined(SYL_DEBUG)
	return((vec1.x * vec2.x) + (vec1.y * vec2.y) + (vec1.z * vec2.z) + (vec1.w * vec2.w));
#else
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_dp_ps(vec1.v, vec2.v, 0xFF);
	return(*(float*)&Result);
#else
	return (vec1.e[0] * vec2.e[0]) + (vec1.e[1] * vec2.e[1]) + (vec1.e[2] * vec2.e[2]) + (vec1.e[3] * vec2.e[3]);
#endif
#endif
}

SYL_INLINE svec4 s_vec4_hadamard(svec4 vec1, svec4 vec2)
{
	svec4 Result = { { (vec1.x * vec2.x), (vec1.y * vec2.y), (vec1.z * vec2.z), (vec1.w * vec2.w) } };
	return(Result);
}

SYL_INLINE float s_vec4_length(svec4 vec1)
{
#if defined(SYL_DEBUG)
	return(sqrt((vec1.x * vec1.x) + (vec1.y * vec1.y) + (vec1.z * vec1.z) + (vec1.w * vec1.w)));
#else
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 V = _mm_load_ps(vec1.e);
	__m128 A1 = _mm_mul_ps(V, V);
	__m128 A2 = _mm_hadd_ps(A1, A1);
	__m128 A3 = _mm_hadd_ps(A2, A2);
	return sqrtf(_mm_cvtss_f32(A3));
#else
	return(sqrt((vec1.x * vec1.x) + (vec1.y * vec1.y) + (vec1.z * vec1.z) + (vec1.w * vec1.w)));
#endif
#endif
}

SYL_INLINE float s_vec4_distance(svec4 vec1, svec4 vec2)
{
	return s_vec4_length(s_vec4_sub(vec1, vec2));
}

SYL_INLINE svec4 s_vec4_normalize(svec4 a)
{
	return(s_vec4_mul_scalar(a, (1.0f / s_vec4_length(a))));
}

SYL_INLINE svec4 s_vec4_lerp(svec4 vec1, svec4 vec2, float t)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_add_ps(vec1.v, _mm_mul_ps((_mm_sub_ps(vec2.v, vec1.v)), _mm_set1_ps(t)));
	return(*(svec4*)&Result);
#else
	svec4 Result = s_vec4_add(vec1, (s_vec4_mul_scalar(s_vec4_sub(vec2, vec1), t)));
	// vec4 Result = vec1 + ((vec2 - vec1) * t);
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4_cross(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 a1 = _SYL_PERMUTE_PS(vec1.v, _SYL_SHUFFLE(3, 0, 2, 1));
	__m128 a2 = _SYL_PERMUTE_PS(vec2.v, _SYL_SHUFFLE(3, 1, 0, 2));
	__m128 r = _mm_mul_ps(a1, a2);
	a1 = _SYL_PERMUTE_PS(a1, _SYL_SHUFFLE(3, 0, 2, 1));
	a2 = _SYL_PERMUTE_PS(a2, _SYL_SHUFFLE(3, 1, 0, 2));
	r = _SYL_ADD_PS(a1, a2, r);
	__m128 re = _mm_and_ps(r, _S_XMM_MASK_3);
	return *(svec4*)&re;
#else
	svec4 Result = { { vec1.e[1] * vec2.e[2] - vec1.e[2] * vec2.e[1],
				   vec1.e[2] * vec2.e[0] - vec1.e[0] * vec2.e[2],
				   vec1.e[0] * vec2.e[1] - vec1.e[1] * vec2.e[0], 0.0f } };
	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4_project(svec4 vectorToProject, svec4 Projectionvector)
{
	float scale = s_vec4_dot(Projectionvector, vectorToProject) / s_vec4_dot(Projectionvector, Projectionvector);
	return(s_vec4_mul_scalar(Projectionvector, scale));
}

/* Per component comparsion to return a vector containing the largest components */
SYL_INLINE svec4 s_vec4_max_vector(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_max_ps(vec1.v, vec2.v);
	return *(svec4*)&Result;
#else
	svec4 Result;
	if (vec1.x > vec2.x)
		Result.x = vec1.x;
	else
		Result.x = vec2.x;

	if (vec1.y > vec2.y)
		Result.y = vec1.y;
	else
		Result.y = vec2.y;

	if (vec1.z > vec2.z)
		Result.z = vec1.z;
	else
		Result.z = vec2.z;

	if (vec1.w > vec2.w)
		Result.w = vec1.w;
	else
		Result.w = vec2.w;

	return(Result);
#endif
}

/* Per component comparsion to return a vector containing the smollest components */
SYL_INLINE svec4 s_vec4_min_vector(svec4 vec1, svec4 vec2)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Result = _mm_min_ps(vec1.v, vec2.v);
	return *(svec4*)&Result;
#else
	svec4 Result;
	if (vec1.x < vec2.x)
		Result.x = vec1.x;
	else
		Result.x = vec2.x;

	if (vec1.y < vec2.y)
		Result.y = vec1.y;
	else
		Result.y = vec2.y;

	if (vec1.z < vec2.z)
		Result.z = vec1.z;
	else
		Result.z = vec2.z;

	if (vec1.w < vec2.w)
		Result.w = vec1.w;
	else
		Result.w = vec2.w;

	return(Result);
#endif
}

SYL_INLINE svec4 s_vec4_clamp(svec4 value, svec4 Min, svec4 max)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	// __m128 Result = _mm_max_ps(Min.v, max.v);
	// Result = _mm_max_ps(max.v, Result);
	// return *(svec4*)&Result;

	__m128 Result = _mm_min_ps(_mm_max_ps(value.v, Min.v), max.v);
	return *(svec4*)&Result;
#else
	svec4 Result = s_vec4_min_vector(s_vec4_max_vector(value, Min), max);
	return(Result);
#endif
}

/* Return the biggest element inside vec4 */
SYL_INLINE float s_vec4_max(svec4 A)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 value = _mm_load_ps(A.e);
	__m128 r = _mm_max_ps(value, value);
	return *(float*)&r;
#else
	/* TODO(hsnovel): Implement ! */
	return(A.e[0]);
#endif
}

/* Return the smollest element inside vec4 */
SYL_INLINE float s_vec4_min(svec4 A)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 value = _mm_load_ps(A.e);
	__m128 r = _mm_min_ps(value, value);
	return *(float*)&r;
#else
	/* TODO(hsnovel): Implement ! */
	return(A.e[0]);
#endif
}

SYL_INLINE float s_vec4_sum(svec4 vec1)
{
	return((vec1.x + vec1.y) + (vec1.z + vec1.w));
}

/*********************************************
 *                 MATRIX 4X4                 *
 *********************************************/

/* Build an identity matrix with given value */
SYL_INLINE smat4 SMAT4(float value)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	smat4 R;
	R.v[0] = _mm_set_ps(value, 0, 0, 0);
	R.v[1] = _mm_set_ps(0, value, 0, 0);
	R.v[2] = _mm_set_ps(0, 0, value, 0);
	R.v[3] = _mm_set_ps(0, 0, 0, value);
	return(R);
#else
	smat4 Result = { { value, 0, 0, 0,
				   0, value, 0, 0,
				   0, 0, value, 0,
				   0, 0, 0, value } };
	return(Result);
#endif
}

SYL_INLINE smat4 SMAT4F(float m00, float m01, float m02, float m03,
		       float m10, float m11, float m12, float m13,
		       float m20, float m21, float m22, float m23,
		       float m30, float m31, float m32, float m33)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	smat4 R;
	R.v[0] = _mm_set_ps(m00, m01, m02, m03);
	R.v[1] = _mm_set_ps(m10, m11, m12, m13);
	R.v[2] = _mm_set_ps(m20, m21, m22, m23);
	R.v[3] = _mm_set_ps(m30, m31, m32, m33);
	return(R);
#else
	smat4 Result = { { m00, m01, m02, m03,
				   m10, m11, m12, m13,
				   m20, m21, m22, m23,
				   m30, m31, m32, m33 } };
	return(Result);
#endif
}

SYL_INLINE smat4 SMAT4A(float* a)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	smat4 R;
	R.v[0] = _mm_set_ps(a[0], a[1], a[2], a[3]);
	R.v[1] = _mm_set_ps(a[4], a[5], a[6], a[7]);
	R.v[2] = _mm_set_ps(a[8], a[9], a[10], a[11]);
	R.v[3] = _mm_set_ps(a[12], a[13], a[14], a[15]);
	return(R);
#else
	smat4 Result = { { a[0], a[1], a[2], a[3],
				   a[4], a[5], a[6], a[7],
				   a[8], a[9], a[10], a[11],
				   a[12], a[13], a[14], a[15] } };
	return(Result);
#endif
}

SYL_INLINE void s_mat4_zero(smat4* Matrix)
{
#if defined(SYL_ENABLE_AVX)
	_mm256_store_ps(Matrix.e, _S_YMM_ZERO);
	_mm256_store_ps(Matrix.e + 8, _S_YMM_ZERO);
#elif defined(SYL_ENABLE_SSE4)
	_mm_store_ps(Matrix->e, _S_XMM_ZERO);
	_mm_store_ps(Matrix->e + 4, _S_XMM_ZERO);
	_mm_store_ps(Matrix->e + 8, _S_XMM_ZERO);
	_mm_store_ps(Matrix->e + 12, _S_XMM_ZERO);
#else
	memset(Matrix, 0, sizeof(float) * 16);
#endif
}

SYL_INLINE smat4 s_mat4_identity()
{
	return(_S_IDENT4X4);
}

SYL_INLINE void s_mat4_identityp(smat4 *ptr)
{
#if defined(SYL_ENABLE_SSE4)
	ptr->v[0] = _mm_set_ps(0, 0, 0, 1);
	ptr->v[1] = _mm_set_ps(0, 0, 1, 0);
	ptr->v[2] = _mm_set_ps(0, 1, 0, 0);
	ptr->v[3] = _mm_set_ps(1, 0, 0, 0);
#else
	memcpy(ptr, &_S_IDENT4X4, sizeof(smat4));
#endif
}

SYL_INLINE bool s_mat4_is_identity(smat4 Mat)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Temp1 = _mm_cmpeq_ps(Mat.v[0], _S_IDENT4x4R0);
	__m128 Temp2 = _mm_cmpeq_ps(Mat.v[1], _S_IDENT4x4R1);
	__m128 Temp3 = _mm_cmpeq_ps(Mat.v[2], _S_IDENT4x4R2);
	__m128 Temp4 = _mm_cmpeq_ps(Mat.v[3], _S_IDENT4x4R3);
	Temp1 = _mm_and_ps(Temp1, Temp2);
	Temp3 = _mm_and_ps(Temp3, Temp4);
	Temp1 = _mm_and_ps(Temp1, Temp3);
	return(_mm_movemask_ps(Temp1) == 0x0f);
#else
	/* Oh god don't use this on non intrinsic version... */
	/* 1278421 cycles incoming...  */
	if (Mat.e[0] == 1.0f && Mat.e[1] == 0.0f && Mat.e[2] == 0.0f && Mat.e[3] == 0.0f &&
	    Mat.e[4] == 0.0f && Mat.e[5] == 1.0f && Mat.e[6] == 0.0f && Mat.e[7] == 0.0f &&
	    Mat.e[8] == 0.0f && Mat.e[9] == 0.0f && Mat.e[10] == 1.0f && Mat.e[11] == 0.0f &&
	    Mat.e[12] == 0.0f && Mat.e[13] == 0.0f && Mat.e[14] == 0.0f && Mat.e[15] == 1.0f) {
		/* Nice you spend couple of frames calculating this... */
		return(true);
	}
	else {
		/* Maybe even failed what a shame.. */
		return(false);
	}
#endif
}

/* Multiply two 4x4 Matricies */
SYL_INLINE smat4 s_mat4_mul(smat4 Matrix1, smat4 Matrix2)
{
#if defined(SYL_ENABLE_AVX)
	mat4 Result;
	__m256 Temp0 = _mm256_castps128_ps256(Matrix1.v[0]);
	Temp0 = _mm256_insertf128_ps(Temp0, Matrix1.v[1], 1);

	__m256 Temp1 = _mm256_castps128_ps256(Matrix1.v[2]);
	Temp1 = _mm256_insertf128_ps(Temp1, Matrix1.v[3], 1);

	__m256 Temp3 = _mm256_castps128_ps256(Matrix2.v[0]);
	Temp3 = _mm256_insertf128_ps(Temp3, Matrix2.v[1], 1);

	__m256 Temp4 = _mm256_castps128_ps256(Matrix2.v[2]);
	Temp4 = _mm256_insertf128_ps(Temp4, Matrix2.v[3], 1);

	__m256 CXA0 = _mm256_shuffle_ps(Temp0, Temp0, _SYL_SHUFFLE(0, 0, 0, 0));
	__m256 CXA1 = _mm256_shuffle_ps(Temp1, Temp1, _SYL_SHUFFLE(0, 0, 0, 0));

	__m256 CXB1 = _mm256_permute2f128_ps(Temp3, Temp3, 0x00);

	__m256 CXC0 = _mm256_mul_ps(CXA0, CXB1);
	__m256 CXC1 = _mm256_mul_ps(CXA1, CXB1);

	CXA0 = _mm256_shuffle_ps(Temp0, Temp0, _SYL_SHUFFLE(1, 1, 1, 1));
	CXA1 = _mm256_shuffle_ps(Temp1, Temp1, _SYL_SHUFFLE(1, 1, 1, 1));

	CXB1 = _mm256_permute2f128_ps(Temp3, Temp3, 0x11);

	__m256 CXC2 = _mm256_fmadd_ps(CXA0, CXB1, CXC0);
	__m256 CXC3 = _mm256_fmadd_ps(CXA1, CXB1, CXC1);

	CXA0 = _mm256_shuffle_ps(Temp0, Temp0, _SYL_SHUFFLE(2, 2, 2, 2));
	CXA1 = _mm256_shuffle_ps(Temp1, Temp1, _SYL_SHUFFLE(2, 2, 2, 2));

	__m256 CXCL1 = _mm256_permute2f128_ps(Temp4, Temp4, 0x00);

	__m256 CXC4 = _mm256_mul_ps(CXA0, CXCL1);
	__m256 CXC5 = _mm256_mul_ps(CXA1, CXCL1);

	CXA0 = _mm256_shuffle_ps(Temp0, Temp0, _SYL_SHUFFLE(3, 3, 3, 3));
	CXA1 = _mm256_shuffle_ps(Temp1, Temp1, _SYL_SHUFFLE(3, 3, 3, 3));

	CXCL1 = _mm256_permute2f128_ps(Temp4, Temp4, 0x11);

	__m256 RXR0 = _mm256_fmadd_ps(CXA0, CXCL1, CXC4);
	__m256 RXR1 = _mm256_fmadd_ps(CXA1, CXCL1, CXC5);

	Temp0 = _mm256_add_ps(CXC2, RXR0);
	Temp1 = _mm256_add_ps(CXC3, RXR1);

	Result.v[0] = _mm256_castps256_ps128(Temp0);
	Result.v[1] = _mm256_extractf128_ps(Temp0, 1);

	Result.v[2] = _mm256_castps256_ps128(Temp1);
	Result.v[3] = _mm256_extractf128_ps(Temp1, 1);

	return(Result);
#elif defined(SYL_ENABLE_SSE4)

	smat4 Result = { 0 };

	__m128 vW = Matrix1.v[0];
	__m128 vX = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(0, 0, 0, 0));
	__m128 vY = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(1, 1, 1, 1));
	__m128 vZ = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(2, 2, 2, 2));

	vW = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(3, 3, 3, 3));
	vX = _mm_mul_ps(vX, Matrix2.v[0]);
	vY = _mm_mul_ps(vY, Matrix2.v[1]);
	vZ = _mm_mul_ps(vZ, Matrix2.v[2]);
	vW = _mm_mul_ps(vW, Matrix2.v[3]);
	vX = _mm_add_ps(vX, vZ);
	vY = _mm_add_ps(vY, vW);
	vX = _mm_add_ps(vX, vY);
	Result.v[0] = vX;

	vW = Matrix1.v[1];
	vX = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(0, 0, 0, 0));
	vY = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(1, 1, 1, 1));
	vZ = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(2, 2, 2, 2));
	vW = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(3, 3, 3, 3));

	vX = _mm_mul_ps(vX, Matrix2.v[0]);
	vY = _mm_mul_ps(vY, Matrix2.v[1]);
	vZ = _mm_mul_ps(vZ, Matrix2.v[2]);
	vW = _mm_mul_ps(vW, Matrix2.v[3]);
	vX = _mm_add_ps(vX, vZ);
	vY = _mm_add_ps(vY, vW);
	vX = _mm_add_ps(vX, vY);

	Result.v[1] = vX;

	vW = Matrix1.v[2];
	vX = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(0, 0, 0, 0));
	vY = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(1, 1, 1, 1));
	vZ = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(2, 2, 2, 2));
	vW = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(3, 3, 3, 3));

	vX = _mm_mul_ps(vX, Matrix2.v[0]);
	vY = _mm_mul_ps(vY, Matrix2.v[1]);
	vZ = _mm_mul_ps(vZ, Matrix2.v[2]);
	vW = _mm_mul_ps(vW, Matrix2.v[3]);
	vX = _mm_add_ps(vX, vZ);
	vY = _mm_add_ps(vY, vW);
	vX = _mm_add_ps(vX, vY);

	Result.v[2] = vX;

	vW = Matrix1.v[3];
	vX = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(0, 0, 0, 0));
	vY = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(1, 1, 1, 1));
	vZ = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(2, 2, 2, 2));
	vW = _SYL_PERMUTE_PS(vW, _SYL_SHUFFLE(3, 3, 3, 3));

	vX = _mm_mul_ps(vX, Matrix2.v[0]);
	vY = _mm_mul_ps(vY, Matrix2.v[1]);
	vZ = _mm_mul_ps(vZ, Matrix2.v[2]);
	vW = _mm_mul_ps(vW, Matrix2.v[3]);

	vX = _mm_add_ps(vX, vZ);
	vY = _mm_add_ps(vY, vW);
	vX = _mm_add_ps(vX, vY);

	Result.v[3] = vX;

	return(Result);
#else
	smat4 Result;
	s_mat4_zero(&Result);

	for (int k = 0; k < 4; ++k) {
		for (int n = 0; n < 4; ++n) {
			for (int i = 0; i < 4; ++i) {
				Result.e2[k][n] += Matrix1.e2[k][i] * Matrix2.e2[i][n];
			}
		}
	}

	return(Result);
#endif
}

SYL_INLINE smat4 s_mat4_transpose(smat4 Mat)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	__m128 Temp1 = _mm_shuffle_ps(Mat.v[0], Mat.v[1], _SYL_SHUFFLE(1, 0, 1, 0));
	__m128 Temp2 = _mm_shuffle_ps(Mat.v[0], Mat.v[1], _SYL_SHUFFLE(3, 2, 3, 2));
	__m128 Temp3 = _mm_shuffle_ps(Mat.v[2], Mat.v[3], _SYL_SHUFFLE(1, 0, 1, 0));
	__m128 Temp4 = _mm_shuffle_ps(Mat.v[2], Mat.v[3], _SYL_SHUFFLE(3, 2, 3, 2));
	smat4 R;
	R.v[0] = _mm_shuffle_ps(Temp1, Temp2, _SYL_SHUFFLE(2, 0, 2, 0));
	R.v[1] = _mm_shuffle_ps(Temp1, Temp2, _SYL_SHUFFLE(3, 1, 3, 1));
	R.v[2] = _mm_shuffle_ps(Temp3, Temp4, _SYL_SHUFFLE(2, 0, 2, 0));
	R.v[3] = _mm_shuffle_ps(Temp3, Temp4, _SYL_SHUFFLE(3, 1, 3, 1));
	return(R);
#else
	smat4 Result;

	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			Result.e2[j][i] = Mat.e2[i][j];
		}
	}
	return(Result);
#endif
}

/* Inverse of a matrix but in the scale of this matrix should be 1 */
SYL_INLINE smat4 s_mat4_inverse_noscale(smat4 Matrix)
{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)

	__m128 Temp0 = _SYL_VEC_SHUFFLE_0101(Matrix.v[0], Matrix.v[1]); // 00, 01, 10, 11
	__m128 Temp1 = _SYL_VEC_SHUFFLE_2323(Matrix.v[0], Matrix.v[1]); // 02, 03, 12, 13

	smat4 Result;
	Result.v[0] = _SYL_VEC_SHUFFLE(Temp0, Matrix.v[2], 0, 2, 0, 3); // 00, 10, 20, 23(=0)
	Result.v[1] = _SYL_VEC_SHUFFLE(Temp0, Matrix.v[2], 1, 3, 1, 3); // 01, 11, 21, 23(=0)
	Result.v[2] = _SYL_VEC_SHUFFLE(Temp1, Matrix.v[2], 0, 2, 2, 3); // 02, 12, 22, 23(=0)

	Result.v[3] = _mm_mul_ps(Result.v[0], _SYL_VEC_SWIZZLE1(Matrix.v[3], 0));
	Result.v[3] = _mm_add_ps(Result.v[3], _mm_mul_ps(Result.v[1], _SYL_VEC_SWIZZLE1(Matrix.v[3], 1)));
	Result.v[3] = _mm_add_ps(Result.v[3], _mm_mul_ps(Result.v[2], _SYL_VEC_SWIZZLE1(Matrix.v[3], 2)));
	Result.v[3] = _mm_sub_ps(_mm_setr_ps(0.f, 0.f, 0.f, 1.f), Result.v[3]);
	return(Result);
#else
	/* TODO(hsnovel): This is temporary to prevent gcc warning, implemenet this later ! */
	return(Matrix);
#endif
}

/* Inverse of a matrix*/
/*
 * SYL_INLINE smat4 Inverse(smat4 Matrix)
 * {
 * #if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
 *     __m128 a0 = _SYL_VEC_SHUFFLE_0101(Matrix.v[0], Matrix.v[1]); // 00, 01, 10, 11
 *     __m128 a1 = _SYL_VEC_SHUFFLE_2323(Matrix.v[0], Matrix.v[1]); // 02, 03, 12, 13
 *
 *     smat4 Result;
 *     Result.v[0] = _SYL_VEC_SHUFFLE(a0, Matrix.v[2], 0, 2, 0, 3); // 00, 10, 20, 23(=0)
 *     Result.v[1] = _SYL_VEC_SHUFFLE(a0, Matrix.v[2], 1, 3, 1, 3); // 01, 11, 21, 23(=0)
 *     Result.v[2] = _SYL_VEC_SHUFFLE(a1, Matrix.v[2], 0, 2, 2, 3); // 02, 12, 22, 23(=0)
 *
 *     __m128 SizeSquared = _mm_mul_ps(Result.v[0], Result.v[0]);
 *     SizeSquared = _mm_add_ps(SizeSquared, _mm_mul_ps(Result.v[1], Result.v[1]));
 *     SizeSquared = _mm_add_ps(SizeSquared, _mm_mul_ps(Result.v[2], Result.v[2]));
 *
 *     __m128 Sqr = _mm_blendv_ps(_mm_div_ps(_S_XMM_ZERO, SizeSquared), _S_XMM_ZERO, _mm_cmplt_ps(SizeSquared, _mm_set1_ps(_SYL_SMALL_NUMBER)));
 *
 *     Result.v[0] = _mm_mul_ps(Result.v[0], Sqr);
 *     Result.v[1] = _mm_mul_ps(Result.v[1], Sqr);
 *     Result.v[2] = _mm_mul_ps(Result.v[2], Sqr);
 *
 *     Result.v[3] = _mm_mul_ps(Result.v[0], _SYL_VEC_SWIZZLE1(Matrix.v[3], 0));
 *     Result.v[3] = _mm_add_ps(Result.v[3], _mm_mul_ps(Result.v[1], _SYL_VEC_SWIZZLE1(Matrix.v[3], 1)));
 *     Result.v[3] = _mm_add_ps(Result.v[3], _mm_mul_ps(Result.v[2], _SYL_VEC_SWIZZLE1(Matrix.v[3], 2)));
 *     Result.v[3] = _mm_sub_ps(_mm_setr_ps(0.f, 0.f, 0.f, 1.f), Result.v[3]);
 *
 *     return(Result);
 * #else
 *     /\* TODO(hsnovel): This is temporary to prevent gcc warning, implemenet this later ! *\/
 *     return(Matrix);
 * #endif
 * }
 */

SYL_INLINE svec4 s_mat4_transform(smat4 Matrix, svec4 vector) {
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	svec4 Result;
	Result.v = _SYL_PERMUTE_PS(vector.v, _MM_SHUFFLE(3, 3, 3, 3));
	Result.v = _mm_mul_ps(Result.v, Matrix.v[3]);
	__m128 Temp = _SYL_PERMUTE_PS(vector.v, _MM_SHUFFLE(2, 2, 2, 2));
	Result.v = _SYL_ADD_PS(Temp, Matrix.v[2], Result.v);
	Temp = _SYL_PERMUTE_PS(vector.v, _MM_SHUFFLE(1, 1, 1, 1));
	Result.v = _SYL_ADD_PS(Temp, Matrix.v[1], Result.v);
	Temp = _SYL_PERMUTE_PS(vector.v, _MM_SHUFFLE(0, 0, 0, 0));
	Result.v = _SYL_ADD_PS(Temp, Matrix.v[0], Result.v);
	return(Result);
#else
	svec4 Result;

	Result.x = vector.x * Matrix.e2[0][0] + vector.y * Matrix.e2[0][1] + vector.z * Matrix.e2[0][2] + vector.w * Matrix.e2[0][3];
	Result.y = vector.x * Matrix.e2[1][0] + vector.y * Matrix.e2[1][1] + vector.z * Matrix.e2[1][2] + vector.w * Matrix.e2[1][3];
	Result.z = vector.x * Matrix.e2[2][0] + vector.y * Matrix.e2[2][1] + vector.z * Matrix.e2[2][2] + vector.w * Matrix.e2[2][3];
	Result.w = vector.x * Matrix.e2[3][0] + vector.y * Matrix.e2[3][1] + vector.z * Matrix.e2[3][2] + vector.w * Matrix.e2[3][3];

	return(Result);
#endif
}

SYL_INLINE svec4 s_mat4_mul_vec4(smat4 Matrix1, svec4 vector)
{
	return s_mat4_transform(Matrix1, vector);
}

SYL_INLINE svec3 s_mat4_mul_vec3(smat4 Matrix1, svec3 vector)
{
	svec4 Vec = s_mat4_transform(Matrix1, SVEC4VF(vector, 1.0f));
	svec3 Result = { { Vec.x, Vec.y, Vec.z } };
	return(Result);
}

SYL_INLINE smat4 s_mat4_translate(smat4 matrix, svec3 vec)
{
	/* TODO: add simd version if I can figure out how to */
	svec4 r1 = s_vec4_mul_scalar(matrix.v4d[0], vec.x);
	svec4 r2 = s_vec4_mul_scalar(matrix.v4d[1], vec.y);
	svec4 r3 = s_vec4_mul_scalar(matrix.v4d[2], vec.z);

	svec4 rf = s_vec4_add(r1, r2);
	rf = s_vec4_add(rf, r3);
	rf = s_vec4_add(rf, matrix.v4d[3]);
	matrix.v4d[3] = rf;

	return matrix;
}

SYL_INLINE smat4 s_mat4_rotate(smat4 matrix, float angle, svec3 vec)
{
	/* NOTE: too many unnecesarry temporary variables, clean it up */
	float c = cos(angle);
	float s = sin(angle);

	svec3 axis = s_vec3_normalize(vec);
	svec3 temp = s_vec3_mul_scalar(axis, (1 - c));

	smat4 rotate;
	rotate.m00 = c + temp.x * axis.x;
	rotate.m01 = temp.x * axis.y + s * axis.z;
	rotate.m02 = temp.x * axis.x - s * axis.y;

	rotate.m10 = temp.y * axis.x - s * axis.z;
	rotate.m11 = c + temp.y * axis.y;
	rotate.m12 = temp.y * axis.z + s * axis.x;

	rotate.m20 = temp.z * axis.x + s * axis.y;
	rotate.m21 = temp.z * axis.y - s * axis.x;
	rotate.m22 = c + temp.z * axis.z;

	smat4 result;

	result.v4d[0] = s_vec4_mul(matrix.v4d[0], SVEC4(rotate.m00, rotate.m00, rotate.m00, rotate.m00));
	result.v4d[0] = s_vec4_add(result.v4d[0], s_vec4_mul(matrix.v4d[1], SVEC4(rotate.m01, rotate.m01, rotate.m01, rotate.m01)));
	result.v4d[0] = s_vec4_add(result.v4d[0], s_vec4_mul(matrix.v4d[2], SVEC4(rotate.m02, rotate.m02, rotate.m02, rotate.m02)));

	svec4 rta = s_vec4_mul(matrix.v4d[0], SVEC4(rotate.m10, rotate.m10, rotate.m10, rotate.m10));
	svec4 rtb = s_vec4_mul(matrix.v4d[1], SVEC4(rotate.m11, rotate.m11, rotate.m11, rotate.m11));
	svec4 rtc = s_vec4_mul(matrix.v4d[2], SVEC4(rotate.m12, rotate.m12, rotate.m12, rotate.m12));
	svec4 tmpr = s_vec4_add(rta, rtb);
	tmpr = s_vec4_add(tmpr, rtc);
	result.v4d[1] = tmpr;

	svec4 rt1 = s_vec4_mul(matrix.v4d[0], SVEC4(rotate.m20, rotate.m20, rotate.m20, rotate.m20));
	svec4 rt2 = s_vec4_mul(matrix.v4d[1], SVEC4(rotate.m21, rotate.m21, rotate.m21, rotate.m21));
	svec4 rt3 = s_vec4_mul(matrix.v4d[2], SVEC4(rotate.m22, rotate.m22, rotate.m22, rotate.m22));
	svec4 tmpa = s_vec4_add(rt1, rt2);
	tmpa = s_vec4_add(tmpa, rt3);
	result.v4d[2] = tmpa;

	result.v4d[3] = matrix.v4d[3];
	return result;
}

SYL_INLINE smat4 s_mat4_xrotation(float Angle)
{
	float CosAngle = cos(Angle);
	float SinAngle = sin(Angle);

	smat4 Result = { {
			1, 0, 0, 0,
			0, CosAngle,-SinAngle, 0,
			0, SinAngle, CosAngle, 0,
			0, 0, 0, 1
		} };

	return(Result);
}

SYL_INLINE smat4 s_mat4_yrotation(float Angle)
{
	float CosAngle = cos(Angle);
	float SinAngle = sin(Angle);

	smat4 Result = { {
			CosAngle, 0, SinAngle, 0,
			0, 1, 0, 0,
			-SinAngle, 0, CosAngle, 0,
			0, 0, 0, 1
		} };

	return(Result);
}

SYL_INLINE smat4 s_mat4_zrotation(float Angle)
{
	float CosAngle = cos(Angle);
	float SinAngle = sin(Angle);

	smat4 result = { {
			CosAngle, -SinAngle, 0, 0,
			SinAngle, CosAngle, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		} };

	return(result);
}

SYL_INLINE smat4 s_mat4_translation(svec3 vector)
{
#if defined(SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	smat4 result;
	result.v[0] = _S_IDENT4x4R0;
	result.v[1] = _S_IDENT4x4R1;
	result.v[2] = _S_IDENT4x4R2;
	result.v[0] = _mm_set_ps(vector.x, vector.y, vector.z, 1.0f);
	return(result);
#else
	smat4 result = { {
			1, 0, 0, vector.x,
			0, 1, 0, vector.y,
			0, 0, 1, vector.z,
			0, 0, 0, 1,
		} };

	return(result);
#endif
}

/* No LH version for now...  */
SYL_INLINE smat4 s_mat4_perspective_projection_rh(float fov, float aspect_ratio, float NearClipPlane, float FarClipPlane)
{
#if defined(SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
	float Sin = sin(0.5f * fov);
	float Cos = cos(0.5f * fov);
	float Range = FarClipPlane / (NearClipPlane - FarClipPlane);
	float Height = Cos / Sin;
	__m128 Mem = { Height / aspect_ratio, Height, Range, Range * NearClipPlane };
	__m128 Mem2 = Mem;
	__m128 Temp = _mm_setzero_ps();
	Temp = _mm_move_ss(Temp, Mem2);
	smat4 result;
	result.v[0] = Temp;
	Temp = Mem2;
	Temp = _mm_and_ps(Temp, _S_XMM_MASK_Y);
	result.v[1] = Temp;
	Temp = _mm_setzero_ps();
	Mem2 = _mm_shuffle_ps(Mem2, _S_IDENT4x4R3, _MM_SHUFFLE(3, 2, 3, 2));
	Temp = _mm_shuffle_ps(Temp, Mem2, _MM_SHUFFLE(3, 0, 0, 0));
	result.v[2] = Temp;
	Temp = _mm_shuffle_ps(Temp, Mem2, _MM_SHUFFLE(2, 1, 0, 0));
	result.v[3] = Temp;
	return(result);
#else
	//float Cotan = 1.0f / tanf(fov / 2.0f);

	//mat4 result = {
	//	Cotan / aspect_ratio, 0.0f, 0.0f, 0.0f,
	//	0.0f, Cotan, 0.0f, 0.0f,
	//	0.0f, 0.0f, (FarClipPlane + NearClipPlane) / (NearClipPlane - FarClipPlane), -1.0f,
	//	0.0f, 0.0f, (2.0f * FarClipPlane * NearClipPlane) / (NearClipPlane - FarClipPlane), 0.0f
	//};

	//return(result);

	float tanHalffovy = tan(fov / 2);

	smat4 result;
	s_mat4_zero(&result);
	result.e2[0][0] = 1 / (aspect_ratio * tanHalffovy);
	result.e2[1][1] = 1 / (tanHalffovy);
	result.e2[2][2] = -(FarClipPlane + NearClipPlane) / (FarClipPlane - NearClipPlane);
	result.e2[2][3] = -1;
	result.e2[3][2] = -(2 * FarClipPlane * NearClipPlane) / (FarClipPlane - NearClipPlane);
	return result;
#endif
}

SYL_INLINE smat4 s_mat4_orthographic_projection_rh(float aspect_ratio, float near_clip_plane, float far_clip_plane)
{
	float Ral = 1.0f;
	float Rsl = aspect_ratio;
	float Fan = near_clip_plane;
	float Fsn = far_clip_plane;
	float Tab = 2.0f / (Fan - Fsn);
	float Tsb = (Fan + Fsn) / (Fan - Fsn);

	smat4 result = { { 1 / Ral,   0,   0,    0,
				   0, 1 / Rsl,   0,    0,
				   0,   0, 1 / Tab, -Tsb / Tab,
				   0,   0,   0,    1 } };
	return(result);
}

#ifdef __cplusplus
_SYL_CPP_EXTERN_END
#endif

#endif // SYL_IMPLEMENTATION
