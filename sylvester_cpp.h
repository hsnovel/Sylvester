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

#include <math.h>
#include <stdbool.h>
#include <assert.h>

#if defined(__clang__)
#	define _SYL_SET_SPEC_ALIGN(x) __attribute__((aligned(x)))

#elif defined(__GNUC__) || defined(__GNUG__)
#	define _SYL_SET_SPEC_ALIGN(x) __attribute__((aligned(x)))

#elif defined(_MSC_VER)
#	define _SYL_SET_SPEC_ALIGN(x) __declspec(align(x))
#else
#	define _SYL_SET_SPEC_ALIGN(x)
#endif

#if defined(SYL_NO_INLINE)
#	define SYL_INLINE static
#elif defined (SYL_NO_STATIC)
#	define SYL_INLINE inline
#else
#	define SYL_INLINE static inline
#endif

#ifndef SYL_NO_STATIC
#define SYL_STATIC
#endif

#define SYL_PI 3.14159265359f

#define _SYL_SHUFFLE(a,b,c,d) (((a) << 6) | ((b) << 4) | \
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

namespace Syl
{
	union vec2
	{
		struct { float x, y; };
		struct { float s; float t; };
		float e[2];
#if defined(SYL_ENABLE_SSE4)
		__m64 v;
#endif
	} _SYL_SET_SPEC_ALIGN(8);

	union vec3
	{
		struct { float x; float y; float z; };
		struct { float r; float g; float b; };
		struct { float s; float t; float p; };
		float e[3];
	};

	union vec4
	{
		struct { float x; float y; float z; float w; };
		struct { float r; float g; float b; float a; };
		struct { float s; float t; float m; float q; };
		float e[4];
#if defined(SYL_ENABLE_SSE4)
		__m128 v;
#endif
	} _SYL_SET_SPEC_ALIGN(16);

	/* We use column-major matricies */
	union mat4
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
		vec4 v4d[4];
#if defined(SYL_ENABLE_SSE4)
		__m128 v[4];
#endif
#if defined(SYL_ENABLE_AVX)
		__m256 v2[2];
#endif
	} _SYL_SET_SPEC_ALIGN(16);

	static mat4 _S_IDENT4X4 = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
	};

	/* TODO(xcatalyst): GCC throws "defined but not used [-Wunused-variable]" */
#if !defined(SYL_ENABLE_SSE4) || !defined(SYL_ENABLE_AVX)
	static mat4 _S_ZERO4X4 = {
	0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f
	};
#endif


#if defined(SYL_ENABLE_SSE4)
	static __m128 _S_XMM_ZERO = { 0.0f, 0.0f, 0.0f, 0.0f };
	static __m128 _S_IDENT4x4R0 = { 1.0f, 0.0f, 0.0f, 0.0f };
	static __m128 _S_IDENT4x4R1 = { 0.0f, 1.0f, 0.0f, 0.0f };
	static __m128 _S_IDENT4x4R2 = { 0.0f, 0.0f, 1.0f, 0.0f };
	static __m128 _S_IDENT4x4R3 = { 0.0f, 0.0f, 0.0f, 1.0f };
	static __m128 _S_XMM_MASK_3 = { (float)0xFFFFFFFF, (float)0xFFFFFFFF, (float)0xFFFFFFFF, (float)0x00000000 };
	static __m128 _S_XMM_MASK_Y = { 0x00000000, (float)0xFFFFFFFF, 0x00000000, 0x00000000 };
#endif
#if defined(SYL_ENABLE_AVX)
	static __m256 _S_YMM_ZERO = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
#endif

	SYL_INLINE float RadianToDegree(float Radian)
	{
		return(Radian * (180 / SYL_PI));
	}

	SYL_INLINE float DegreeToRadian(float Degree)
	{
		return (Degree * (SYL_PI / 180));
	}

	SYL_INLINE float Round(float A)
	{
		return((int)(A + 0.5f));
	}

	SYL_INLINE double Round(double A)
	{
		return((int)(A + 0.5f));
	}

	SYL_INLINE float Ceil(float A)
	{
		return((int)(A + 1.0f));
	}

	SYL_INLINE double Ceil(double A)
	{
		return((int)(A + 1.0f));
	}

	SYL_INLINE float Floor(float A)
	{
		return((int)A);
	}

	SYL_INLINE double Floor(double A)
	{
		return((int)A);
	}

	/* Unpack four 8-bit BGRA values into vec4 */
	SYL_INLINE vec4 BGRAUnpack(int Color)
	{
		vec4 Result = { {
			(float)((Color >> 16) & 0xFF),
			(float)((Color >> 8) & 0xFF),
			(float)((Color >> 0) & 0xFF),
			(float)((Color >> 24) & 0xFF)
			} };
		return(Result);
	}

	/* Pack four 8-bit RGB values */
	SYL_INLINE unsigned int BGRAPack(vec4 Color)
	{
		unsigned int Result =
			((unsigned int)(Color.a) << 24) |
			((unsigned int)(Color.r) << 16) |
			((unsigned int)(Color.g) << 8) |
			((unsigned int)(Color.b) << 0);
		return(Result);
	}

	/* Unpack four 8-bit RGBA values into vec4 */
	SYL_INLINE vec4 RGBAUnpack(unsigned int Color)
	{
		vec4 Result = {
			(float)((Color >> 24) & 0xFF),
			(float)((Color >> 16) & 0xFF),
			(float)((Color >> 8) & 0xFF),
			(float)((Color >> 0) & 0xFF)
		};
		return(Result);
	}

	/* Pack four 8-bit RGBA values */
	SYL_INLINE unsigned int RGBAPack(vec4 Color)
	{
		unsigned int Result =
			((unsigned int)(Color.a) << 24) |
			((unsigned int)(Color.b) << 16) |
			((unsigned int)(Color.g) << 8) |
			((unsigned int)(Color.r) << 0);

		return(Result);

	}

	SYL_INLINE float Clamp(float Value, float Min, float Max)
	{
		float Result = Value;

		if (Result < Min)
		{
			Result = Min;
		}
		else if (Result > Max)
		{
			Result = Max;
		}

		return(Result);
	}

	SYL_INLINE double Clampd(double Value, double Min, double Max)
	{
		double Result = Value;

		if (Result < Min)
		{
			Result = Min;
		}
		else if (Result > Max)
		{
			Result = Max;
		}

		return(Result);
	}

	SYL_INLINE int Clamp(int Value, int Min, int Max)
	{
		int Result = Value;

		if (Result < Min)
		{
			Result = Min;
		}
		else if (Result > Max)
		{
			Result = Max;
		}

		return(Result);
	}

	SYL_INLINE float Clamp01(float Value)
	{
		return(Clamp(Value, 0.0f, 1.0f));
	}

	SYL_INLINE double Clamp01(double Value)
	{
		return(Clamp(Value, 0.0f, 1.0f));
	}

	SYL_INLINE float ClampAboveZero(float Value)
	{
		return (Value < 0) ? 0.0f : Value;
	}

	SYL_INLINE float ClampBelowZero(float Value)
	{
		return (Value > 0) ? 0.0f : Value;
	}

	SYL_INLINE bool IsInRange(float Value, float Min, float Max)
	{
		return(((Min <= Value) && (Value <= Max)));
	}

	SYL_INLINE float Lerp(float A, float t, float B)
	{
		return (1.0f - t) * A + t * B;
	}

	/* It is handy to type square if the expression happens to be very long */
	SYL_INLINE float Square(float x)
	{
		return(x * x);
	}

	/* Absoule value */
	SYL_INLINE float Abs(float x)
	{
		return *((unsigned int*)(&x)) &= 0xffffffff >> 1;
	}

	/* Find the hypotenuse of a triangle given two other sides */
	SYL_INLINE float Pythagorean(float x, float y)
	{
		return sqrt(x * x + y * y);
	}

	/* Maximum of two values */
	SYL_INLINE float Max(float x, float y)
	{
		if (x > y)
			return(x);
		return(y);
	}

	SYL_INLINE int Max(int x, int y)
	{
		if (x > y)
			return(x);
		return(y);
	}

	SYL_INLINE int Min(int x, int y)
	{
		if (x < y)
			return(x);
		return(y);
	}

	SYL_INLINE float Min(float x, float y)
	{
		if (x < y)
			return(x);
		return(y);
	}

	SYL_INLINE float Mod(float x, float y)
	{
		return(x - (Round(x / y) * y));
	}

	SYL_INLINE float Pow(float Value, float Times)
	{
		float pow = 1;
		for (int i = 0; i < Times; i++) {
			pow = pow * Value;
		}
		return(pow);
	}

	/* Places -> How many digits you want to keep remained */
	SYL_INLINE float Truncate(float Value, float Remain)
	{
		int Remove = Pow(10, Remain);
		return(Round(Value * Remove) / Remove);
	}

	SYL_INLINE double Truncate(double Value, double Places)
	{
		int Remove = pow(10, Places);
		return(Round(Value * Remove) / Remove);
	}

	SYL_INLINE float Normalize(float Value, float Min, float Max)
	{
		return (Value - Min) / (Max - Min);
	}

	SYL_INLINE float Map(float Value, float SourceMin, float SourceMax, float DestMin, float DestMax)
	{
		return Lerp(Normalize(Value, SourceMin, SourceMax), DestMin, DestMax);
	}

	SYL_INLINE vec3 RGBToHSV(vec3 RGB)
	{
		/* Range the values between 1 and 0*/
		RGB.r = RGB.r / 255.0;
		RGB.g = RGB.g / 255.0;
		RGB.b = RGB.b / 255.0;

		float MaxValue = Max(RGB.r, Max(RGB.g, RGB.b));
		float MinValue = Max(RGB.r, Min(RGB.g, RGB.b));
		float Dif = MaxValue - MinValue;

		vec3 Result;
		Result.x = -1, Result.y = -1;

		if (MaxValue == MinValue)
			Result.x = 0;

		else if (MaxValue == RGB.r)
			Result.x = Mod(60 * ((RGB.g - RGB.b) / Dif) + 360, 360);

		else if (MaxValue == RGB.g)
			Result.x = Mod(60 * ((RGB.b - RGB.r) / Dif) + 120, 360);

		else if (MaxValue == RGB.b)
			Result.x = Mod(60 * ((RGB.r - RGB.g) / Dif) + 240, 360);

		if (MaxValue == 0)
			Result.y = 0;
		else
			Result.y = (Dif / MaxValue) * 100;

		Result.z = MaxValue * 100;
		return(Result);
	}

	/*********************************************
	*                 VECTOR 2D		             *
	*********************************************/

	SYL_INLINE vec2 VEC2(float a, float b)
	{
		vec2 r = { a, b };
		return(r);
	}

	SYL_INLINE vec2 VEC2(float* a)
	{
		vec2 r = { a[0], a[1] };
		return(r);
	}

	SYL_INLINE void Zero(vec2& Vector)
	{
		Vector = {};
	}

	SYL_INLINE bool operator==(vec2 Vec1, vec2 Vec2)
	{
		bool result = false;
		if (Vec1.x == Vec2.x && Vec1.y == Vec2.y)
			result = true;
		return(result);
	}

	SYL_INLINE bool operator==(vec2 Vec1, float Value)
	{
		bool result = false;
		if (Vec1.x == Value && Vec1.y == Value)
			result = true;
		return(result);
	}

	SYL_INLINE bool operator!=(vec2 Vec1, vec2 Vec2)
	{
		bool result = false;
		if (Vec1.x != Vec2.x && Vec1.y != Vec2.y)
			result = true;
		return(result);
	}

	SYL_INLINE bool operator!=(vec2 Vec1, float Value)
	{
		bool result = false;
		if (Vec1.x != Value && Vec1.y != Value)
			result = true;
		return(result);
	}

	SYL_INLINE bool operator>=(vec2 Vec1, vec2 Vec2)
	{
		bool result = false;
		if (Vec1.x > Vec2.x && Vec1.y > Vec2.y)
			result = true;
		return(result);
	}

	SYL_INLINE bool operator>=(vec2 Vec1, float Value)
	{
		bool result = false;
		if (Vec1.x > Value && Vec1.y > Value)
			result = true;
		return(result);
	}

	SYL_INLINE bool operator<=(vec2 Vec1, vec2 Vec2)
	{
		bool result = false;
		if (Vec1.x < Vec2.x && Vec1.y < Vec2.y)
			result = true;
		return(result);
	}

	SYL_INLINE bool operator<=(vec2 Vec1, float Value)
	{
		bool result = false;
		if (Vec1.x < Value && Vec1.y < Value)
			result = true;
		return(result);
	}

	SYL_INLINE vec2 operator+(vec2 Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2));
		return *(vec2*)&r;
#else
		vec2 Result = { (Vec1.x + Vec2.x), (Vec1.y + Vec2.y) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator+=(vec2& Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x + Vec2.x), (Vec1.y + Vec2.y) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec2 operator+(vec2 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_set1_ps(Value));
		return *(vec2*)&r;
#else
		vec2 Result = { (Vec1.x + Value), (Vec1.y + Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator+=(vec2& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_set1_ps(Value));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { Vec1.x + Value, Vec1.y + Value };
		return(Vec1);
#endif
	}

	SYL_INLINE vec2 operator-(vec2 Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2));
		return *(vec2*)&r;
#else
		vec2 Result = { (Vec1.x - Vec2.x), (Vec2.x - Vec2.y) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator-=(vec2& Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x - Vec2.x), (Vec1.y - Vec2.y) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec2 operator-(vec2 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_set1_ps(Value));
		return *(vec2*)&r;
#else
		vec2 Result = { (Vec1.x - Value), (Vec1.y - Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator-=(vec2& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_set1_ps(Value));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x - Value), (Vec1.y - Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec2 operator-(float Value, vec2 Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_set1_ps(Value), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1));
		return *(vec2*)&r;
#else
		vec2 Result = { (Value - Vec1.x), (Value - Vec1.y) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator-=(float Value, vec2& Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_set1_ps(Value), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { (Value - Vec1.x), (Value - Vec1.y) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec2 operator*(vec2 Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2));
		return *(vec2*)&r;
#else
		vec2 Result = { (Vec1.x * Vec2.x), (Vec1.y * Vec2.y) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator*=(vec2& Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x * Vec2.x), (Vec1.y * Vec2.y) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec2 operator*(vec2 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_set1_ps(Value));
		return *(vec2*)&r;
#else
		vec2 Result = { (Vec1.x * Value), (Vec1.y * Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator*=(vec2& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_set1_ps(Value));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x * Value), (Vec1.y * Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec2 operator/(vec2 Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2));
		return *(vec2*)&r;
#else
		vec2 Result = { (Vec1.x / Vec2.x), (Vec1.y / Vec2.y) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator/=(vec2& Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x / Vec2.x), (Vec1.y / Vec2.y) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec2 operator/(vec2 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_set1_ps(Value));
		return *(vec2*)&r;
#else
		vec2 Result = { (Vec1.x / Value), (Vec1.y / Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator/=(vec2& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_set1_ps(Value));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x / Value), (Vec1.y / Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec2 operator/(float Value, vec2 Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_set1_ps(Value), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1));
		return *(vec2*)&r;
#else
		vec2 Result = { (Value / Vec1.x), (Value / Vec1.y) };
		return(Result);
#endif
	}

	SYL_INLINE vec2& operator/=(float Value, vec2& Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_set1_ps(Value), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1));
		Vec1 = *(vec2*)&r;
		return(Vec1);
#else
		Vec1 = { (Value / Vec1.x), (Value / Vec1.y) };
		return(Vec1);
#endif
	}

	/* Negate all components of the vector */
	SYL_INLINE vec2 Negate(vec2 a)
	{
		vec2 Result = { -a.x, -a.y };
		return(Result);
	}

	SYL_INLINE vec2 Floor(vec2 A)
	{
		vec2 Result = { Floor(A.x), Floor(A.y) };
		return(Result);
	}

	/* Round all components of the vec3 to nearest integer*/
	SYL_INLINE vec2 Round(vec2 A)
	{
		vec2 Result = { Round(A.x), Round(A.y) };
		return(Result);
	}

	SYL_INLINE float Dot(vec2 Vec1, vec2 Vec2)
	{
		return((Vec1.x * Vec2.x) + (Vec1.y * Vec2.y));
	}

	SYL_INLINE vec2 Hadamard(vec2 Vec1, vec2 Vec2)
	{
		vec2 Result = { (Vec1.x * Vec2.x), (Vec1.y * Vec2.y) };
		return(Result);
	}

	SYL_INLINE vec2 Lerp(vec2 Vec1, vec2 Vec2, float t)
	{
		vec2 r = { { Vec1.x + (Vec2.x - Vec1.x * t), Vec1.y + ((Vec2.y - Vec1.y * t)) } };
		return r;
	}

	SYL_INLINE vec2 Clamp(vec2 Value, vec2 Min, vec2 Max)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_min_ps(_mm_max_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Value.v), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Min.v)), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Max.v));
		return *(vec2*)&Result;
#else
		vec2 Result = { Clamp(Value.x, Min.x, Max.x), Clamp(Value.y, Min.y, Max.y) };
		return(Result);
#endif
	}

	SYL_INLINE float Length(vec2 Vec1)
	{
		return(sqrt((Vec1.x * Vec1.x) + (Vec1.y * Vec1.y)));
	}

	SYL_INLINE float Distance(vec2 Vec1, vec2 Vec2)
	{
		return Length(Vec1 - Vec2);
	}

	SYL_INLINE vec2 Normalize(vec2 a)
	{
		return(a * (1.0f / Length(a)));
	}

	/* Reflect a position to a normal plane */
	SYL_INLINE vec2 Reflect(vec2 Pos, vec2 N)
	{
		vec2 Normal = Normalize(N);
		float r = Dot(Pos, Normal);
		vec2 t = { (float)(Pos.e[0] - Normal.e[0] * 2.0 * r), (float)(Pos.e[1] - Normal.e[1] * 2.0 * r) };
		return(t);
	}

	/* Project from a position along a vector on to a plane */
	SYL_INLINE vec2 Project(vec2 VectorToProject, vec2 ProjectionVector)
	{
		float scale = Dot(ProjectionVector, VectorToProject) / Dot(ProjectionVector, ProjectionVector);
		return(ProjectionVector * scale);
	}

	/* Flattens a position to a normal plane */
	static inline vec2 Flatten(vec2 Pos, vec2 Normal)
	{
		float f = Dot(Pos, Normal);
		vec2 result = { (Pos.e[0] - Normal.e[0] * f), (Pos.e[1] - Normal.e[1] * f) };
		return(result);
	}

	/* Per component comparsion to return a vector containing the largest components */
	SYL_INLINE vec2 MaxVector(vec2 Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_max_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1.e), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2.e));
		return *(vec2*)&r;
#else
		vec2 Result;
		if (Vec1.x > Vec2.x)
		{
			Result.x = Vec1.x;
		}
		else
		{
			Result.x = Vec2.x;
		}

		if (Vec1.y > Vec2.y)
		{
			Result.y = Vec1.y;
		}
		else
		{
			Result.y = Vec2.y;
		}
		return(Result);
#endif
	}

	/* Per component comparsion to return a vector containing the smollest components */
	SYL_INLINE vec2 MinVector(vec2 Vec1, vec2 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_min_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1.e), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2.e));
		return *(vec2*)&r;
#else
		vec2 Result;
		if (Vec1.x < Vec2.x)
		{
			Result.x = Vec1.x;
		}
		else
		{
			Result.x = Vec2.x;
		}

		if (Vec1.y < Vec2.y)
		{
			Result.y = Vec1.y;
		}
		else
		{
			Result.y = Vec2.y;
		}
		return(Result);

#endif
	}

	/* Return the biggest element inside vec4 */
	SYL_INLINE float Max(vec2 A)
	{
		if (A.e[0] > A.e[1])
		{
			return(A.e[0]);
		}
		else
		{
			return(A.e[1]);
		}
	}

	/* Return the smollest element inside vec4 */
	SYL_INLINE float Min(vec2 A)
	{
		if (A.e[0] < A.e[1])
		{
			return(A.e[0]);
		}
		else
		{
			return(A.e[1]);
		}
	}

	/* Add all components of the vector together */
	SYL_INLINE float Sum(vec2 Vec1)
	{
		return(Vec1.x + Vec1.y);
	}

	SYL_INLINE float AreaOfTriangle(vec2 Vec1, vec2 Vec2, vec2 Vec3)
	{
		float r = ((Vec1.x * Vec2.y) + (Vec2.x * Vec3.y) + (Vec3.x * Vec1.y) - (Vec1.y * Vec2.x) - (Vec2.y * Vec3.x) - (Vec3.y * Vec1.x)) / 2;
		if (r < 0)
		{
			r = -r;
		}
		return(r);
	}

	/*********************************************
	*                 VECTOR 3D					 *
	*********************************************/


	SYL_INLINE vec3 VEC3(float a, float b, float c)
	{
		vec3 r = { a, b, c };
		return(r);
	}

	SYL_INLINE vec3 VEC3(float* a)
	{
		vec3 r = { a[0], a[1], a[2] };
		return(r);
	}

	SYL_INLINE void Zero(vec3& Vector)
	{
		Vector = {};
	}

	SYL_INLINE bool operator==(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpeq_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x == Vec2.x) && (Vec1.y == Vec2.y) && (Vec1.z == Vec2.z))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator==(vec3 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpeq_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x == Value) && (Vec1.y == Value) && (Vec1.z == Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}

#endif
	}

	SYL_INLINE bool operator!=(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)

		__m128 Result = _mm_cmpeq_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return (((_mm_movemask_ps(Result) & 7) == 7) == 0);
#else
		if ((Vec1.x != Vec2.x) && (Vec1.y != Vec2.y) && (Vec1.z == Vec2.z))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator!=(vec3 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpneq_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x != Value) && (Vec1.y != Value) && (Vec1.z == Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator>(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)

		__m128 Result = _mm_cmpgt_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x > Vec2.x) && (Vec1.y > Vec1.y) && (Vec1.z > Vec1.z))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator<(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmplt_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x < Vec2.x) && (Vec1.y < Vec2.y) && (Vec1.z < Vec2.z))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator<(vec3 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmplt_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x < Value) && (Vec1.y < Value) && (Vec1.z < Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}


	SYL_INLINE bool operator>=(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpge_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x >= Vec2.x) && (Vec1.y >= Vec2.y) && (Vec1.z > Vec2.z))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator>=(vec3 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpge_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x >= Value) && (Vec1.y >= Value) && (Vec1.z > Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator<=(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmple_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x <= Vec2.x) && (Vec1.y <= Vec2.y) && (Vec1.z > Vec2.z))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator<=(vec3 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmple_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return (((_mm_movemask_ps(Result) & 7) == 7) != 0);
#else
		if ((Vec1.x <= Value) && (Vec1.y <= Value) && (Vec1.z > Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE vec3 operator+(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(_mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec1), _mm_loadl_pi(_mm_setzero_ps(), (__m64*) & Vec2));
		return *(vec3*)&r;
#else
		vec3 Result = { (Vec1.x + Vec2.x), (Vec1.y + Vec2.y), (Vec1.z + Vec2.z) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator+=(vec3& Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x + Vec2.x), (Vec1.y + Vec2.y), (Vec1.z + Vec2.z) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec3 operator+(vec3 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return *(vec3*)&r;
#else
		vec3 Result = { (Vec1.x + Value), (Vec1.y + Value), (Vec1.z + Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator+=(vec3& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x + Value), (Vec1.y + Value), (Vec1.z + Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec3 operator-(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return *(vec3*)&r;
#else
		vec3 Result = { (Vec1.x - Vec2.x), (Vec1.y - Vec2.y), (Vec1.z - Vec2.z) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator-=(vec3& Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x - Vec2.x), (Vec1.y - Vec2.y), (Vec1.z - Vec2.z) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec3 operator-(vec3 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return *(vec3*)&r;
#else
		vec3 Result = { (Vec1.x - Value), (Vec1.y - Value), (Vec1.z - Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator-=(vec3& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x - Value), (Vec1.y - Value), (Vec1.z - Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec3 operator-(float Value, vec3 Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_set1_ps(Value), _mm_load_ps(Vec1.e));
		return *(vec3*)&r;
#else
		vec3 Result = { (Value - Vec1.x), (Value - Vec1.y), (Value - Vec1.z) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator-=(float Value, vec3& Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_set1_ps(Value), _mm_load_ps(Vec1.e));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Value - Vec1.x), (Value - Vec1.y), (Value - Vec1.z) };
		return(Vec1);
#endif
	}


	SYL_INLINE vec3 operator*(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return *(vec3*)&r;
#else
		vec3 Result = { (Vec1.x * Vec2.x), (Vec1.y * Vec2.y), (Vec1.z * Vec2.z) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator*=(vec3& Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x * Vec2.x), (Vec1.y * Vec2.y), (Vec1.z * Vec2.z) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec3 operator*(vec3 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return *(vec3*)&r;
#else
		vec3 Result = { (Vec1.x * Value), (Vec1.y * Value),  (Vec1.z * Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator*=(vec3& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x * Value), (Vec1.y * Value), (Vec1.z * Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec3 operator/(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return *(vec3*)&r;
#else
		vec3 Result = { (Vec1.x / Vec2.x), (Vec1.y / Vec2.y), (Vec1.z / Vec2.z) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator/=(vec3& Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x / Vec2.x), (Vec1.y / Vec2.y), (Vec1.z / Vec2.z) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec3 operator/(vec3 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return *(vec3*)&r;
#else
		vec3 Result = { (Vec1.x / Value), (Vec1.y / Value), (Vec1.z / Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator/=(float Value, vec3& Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x / Value), (Vec1.y / Value), (Vec1.z / Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec3 operator/(float Value, vec3 Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_set1_ps(Value), _mm_load_ps(Vec1.e));
		return *(vec3*)&r;
#else
		vec3 Result = { (Value / Vec1.x), (Value / Vec1.y), (Value / Vec1.z) };
		return(Result);
#endif
	}

	SYL_INLINE vec3& operator/=(vec3& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		Vec1 = *(vec3*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x / Value), (Vec1.y / Value), (Vec1.z / Value) };
		return(Vec1);
#endif
	}


	SYL_INLINE vec3 Floor(vec3 A)
	{
		vec3 Result = { Floor(A.x), Floor(A.y), Floor(A.z) };
		return(Result);
	}

	/* Round all components of the vec3 to nearest integer*/
	SYL_INLINE vec3 Round(vec3 A)
	{
		vec3 Result = { Round(A.x), Round(A.y), Round(A.z) };
		return(Result);
	}

	/* Negate all components of the vector */
	SYL_INLINE vec3 Negate(vec3 a)
	{
		vec3 r = { -a.x, -a.y, -a.z };
		return(r);
	}

	SYL_INLINE float Dot(vec3 Vec1, vec3 Vec2)
	{
		return((Vec1.x * Vec2.x) + (Vec1.y * Vec2.y) + (Vec1.z * Vec2.z));
	}

	SYL_INLINE vec3 Hadamard(vec3 Vec1, vec3 Vec2)
	{
		vec3 Result = { (Vec1.x * Vec2.x), (Vec1.y * Vec2.y), (Vec1.z * Vec1.z) };
		return(Result);
	}

	SYL_INLINE float Length(vec3 Vec1)
	{
		return(sqrt((Vec1.x * Vec1.x) + (Vec1.y * Vec1.y) + (Vec1.z * Vec1.z)));
	}

	SYL_INLINE float Distance(vec3 Vec1, vec3 Vec2)
	{
		return Length(Vec1 - Vec2);
	}

	SYL_INLINE vec3 Normalize(vec3 a)
	{
		return(a * (1.0f / Length(a)));
	}

	/* Return the biggest element inside vec4 */
	SYL_INLINE float MaxValue(vec3 A)
	{
		/* NOTE(xcatalyst): Current SSE4 version doesn't work. It is removed for now. Reimplement it later.*/
		if (A.e[0] >= A.e[1] && A.e[0] >= A.e[2])
			return(A.e[0]);
		if (A.e[1] >= A.e[0] && A.e[1] >= A.e[2])
			return(A.e[1]);
		if (A.e[2] >= A.e[0] && A.e[2] >= A.e[1])
			return(A.e[2]);

		return(A.e[0]); // HMMMMMMMMMMMMMMMMMMMM
	}

	/* Return the smollest element inside vec4 */
	SYL_INLINE float MinValue(vec3 A)
	{
		/* NOTE(xcatalyst): Current SSE4 version doesn't work. It is removed for now. Reimplement it later.*/
		if (A.e[0] <= A.e[1] && A.e[0] <= A.e[2])
			return(A.e[0]);
		if (A.e[1] <= A.e[0] && A.e[1] <= A.e[2])
			return(A.e[1]);
		if (A.e[2] <= A.e[0] && A.e[2] <= A.e[1])
			return(A.e[2]);

		return(A.e[0]); // HMMMMMMMMMMMMMMMMMMMM
	}

	/* Per component comparsion to return a vector containing the largest components */
	SYL_INLINE vec3 MaxVector(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_max_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return *(vec3*)&r;
#else
		vec3 Result;
		if (Vec1.x > Vec2.x)
		{
			Result.x = Vec1.x;
		}
		else
		{
			Result.x = Vec2.x;
		}

		if (Vec1.y > Vec2.y)
		{
			Result.y = Vec1.y;
		}
		else
		{
			Result.y = Vec2.y;
		}

		if (Vec1.z > Vec2.z)
		{
			Result.z = Vec1.z;
		}
		else
		{
			Result.z = Vec2.z;
		}

		return(Result);
#endif
	}

	/* Per component comparsion to return a vector containing the smollest components */
	SYL_INLINE vec3 MinVector(vec3 Vec1, vec3 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_min_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return *(vec3*)&r;
#else
		vec3 Result;
		if (Vec1.x < Vec2.x)
		{
			Result.x = Vec1.x;
		}
		else
		{
			Result.x = Vec2.x;
		}

		if (Vec1.y < Vec2.y)
		{
			Result.y = Vec1.y;
		}
		else
		{
			Result.y = Vec2.y;
		}

		if (Vec1.z < Vec2.z)
		{
			Result.z = Vec1.z;
		}
		else
		{
			Result.z = Vec2.z;
		}

		return(Result);
#endif
	}

	SYL_INLINE vec3 Clamp(vec3 Value, vec3 Min, vec3 Max)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_max_ps(_mm_setr_ps(Min.x, Min.y, Min.z, 0.0f), _mm_setr_ps(Value.x, Value.y, Value.z, 0.0f));
		Result = _mm_min_ps(_mm_setr_ps(Max.x, Max.y, Max.z, 0.0f), Result);
		return *(vec3*)&Result;
#else
		vec3 Result = MinVector(MaxVector(Value, Min), Max);
		return(Result);
#endif
	}

	/* TODO(xcatalyst): Faster ! */
	SYL_INLINE vec3 Lerp(vec3 Vec1, vec3 Vec2, float t)
	{
		vec3 Result = { Vec1.e[0] + ((Vec2.e[0] - Vec1.e[0]) * t),
				 Vec1.e[1] + ((Vec2.e[1] - Vec1.e[1]) * t),
				 Vec1.e[2] + ((Vec2.e[2] - Vec1.e[2]) * t) };
		return(Result);

	}

	SYL_INLINE vec3 Project(vec3 VectorToProject, vec3 ProjectionVector)
	{
		float scale = Dot(ProjectionVector, VectorToProject) / Dot(ProjectionVector, ProjectionVector);
		return(ProjectionVector * scale);
	}

	SYL_INLINE vec3 Cross(vec3 Vec1, vec3 Vec2)
	{
#if defined(SYL_DEBUG)
		vec3 Result;
		Result.x = Vec1.y * Vec2.z - Vec1.z * Vec2.y;
		Result.y = Vec1.z * Vec2.x - Vec1.x * Vec2.z;
		Result.z = Vec1.x * Vec2.y - Vec1.y * Vec2.x;
		return(Result);
#else
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 V1 = _mm_load_ps(Vec1.e);
		__m128 V2 = _mm_load_ps(Vec2.e);
		__m128 A1 = _SYL_PERMUTE_PS(V1, _SYL_SHUFFLE(3, 0, 2, 1));
		__m128 A2 = _SYL_PERMUTE_PS(V2, _SYL_SHUFFLE(3, 1, 0, 2));
		__m128 r = _mm_mul_ps(A1, A2);
		A1 = _SYL_PERMUTE_PS(A1, _MM_SHUFFLE(3, 0, 2, 1));
		A2 = _SYL_PERMUTE_PS(A2, _MM_SHUFFLE(3, 1, 0, 2));
		r = _SYL_ADD_PS(A1, A2, r);
		__m128 re = _mm_and_ps(r, _S_XMM_MASK_3);
		return *(vec3*)&re;
#else
		vec3 Result = { Vec1.e[1] * Vec2.e[2] - Vec1.e[2] * Vec2.e[1],
				 Vec1.e[2] * Vec2.e[0] - Vec1.e[0] * Vec2.e[2],
				 Vec1.e[0] * Vec2.e[1] - Vec1.e[1] * Vec2.e[0] };
		return(Result);
#endif
#endif
	}

	/* TODO(xcatalyst): Typed this out of my mind, might be incorrect
	check the function later. Currently undocumented. */
	SYL_INLINE float Slope(vec3 PointA, vec3 PointB)
	{
		float Adjacent = sqrt(Square(PointA.z - PointB.z) + Square(PointA.x - PointB.x));
		float Opposite = Abs(PointA.y - PointB.y);
		return(tanf(Opposite / Adjacent));
	}

	//SYL_INLINE float AreaOfTriangle(vec3 a, vec3 b, vec3 c)
	//{
	//	vec3 Vector = a - b;
	//	float Area = sqrt(Dot(Vector, Vector));
	//	vec3 Side = c - b;
	//	float f = Dot(Vector, Side);
	//	Side -= Vector * f;
	//	Area *= sqrtf(Dot(Side, Side));
	//	return(Area);
	//}

	/*********************************************
	*                   VECTOR 4D		         *
	*********************************************/

	SYL_INLINE vec4 VEC4(float a, float b, float c, float d)
	{
		vec4 r = { a, b, c, d };
		return(r);
	}

	SYL_INLINE vec4 VEC4(float* a)
	{
		vec4 r = { a[0], a[1], a[2], a[3] };
		return(r);
	}

	SYL_INLINE vec4 VEC4(vec3 Vector, float Value)
	{
		vec4 r = { Vector.x, Vector.y, Vector.z, Value };
		return(r);
	}

	SYL_INLINE void Zero(vec4& Vector)
	{
#if defined(SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		_mm_store_ps(Vector.e, _S_XMM_ZERO);
#else
		Vector = {};
#endif
	}

	SYL_INLINE bool operator==(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpeq_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x == Vec2.x) && (Vec1.y == Vec2.y) && (Vec1.z == Vec2.z))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator==(vec4 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpeq_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x == Value) && (Vec1.y == Value) && (Vec1.z == Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator!=(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpneq_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x != Vec2.x) && (Vec1.y != Vec2.y) && (Vec1.z != Vec2.z))
		{
			return(true);
		}
		else
		{
			return(false);
		}

#endif
	}

	SYL_INLINE bool operator!=(vec4 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpneq_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x != Value) && (Vec1.y != Value) && (Vec1.z != Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator>(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpgt_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x > Vec2.x) && (Vec1.y > Vec2.y) && (Vec1.z > Vec2.y))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator<(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmplt_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x < Vec2.x) && (Vec1.y < Vec2.y) && (Vec1.z < Vec2.y))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator<(vec4 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmplt_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x < Value) && (Vec1.y < Value) && (Vec1.z < Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}


	SYL_INLINE bool operator>=(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpge_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x >= Vec2.x) && (Vec1.y >= Vec2.y) && (Vec1.z >= Vec2.y))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator>=(vec4 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmpge_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x >= Value) && (Vec1.y >= Value) && (Vec1.z >= Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE bool operator<=(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmple_ps(_mm_load_ps(Vec1.e), _mm_load_ps(Vec2.e));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x <= Vec2.x) && (Vec1.y <= Vec2.y) && (Vec1.z <= Vec2.y))
		{
			return(true);
		}
		else
		{
			return(false);
		}

#endif
	}

	SYL_INLINE bool operator<=(vec4 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_cmple_ps(_mm_load_ps(Vec1.e), _mm_set1_ps(Value));
		return ((_mm_movemask_ps(Result) == 0x0f) != 0);
#else
		if ((Vec1.x <= Value) && (Vec1.y <= Value) && (Vec1.z <= Value))
		{
			return(true);
		}
		else
		{
			return(false);
		}
#endif
	}

	SYL_INLINE vec4 operator+(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(Vec1.v, Vec2.v);
		return *(vec4*)&r;
#else
		vec4 Result = { (Vec1.x + Vec2.x), (Vec1.y + Vec2.y), (Vec1.z + Vec2.z), (Vec1.w + Vec2.w) };
		return(Result);
#endif
	}

	SYL_INLINE vec4& operator+=(vec4& Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(Vec1.v, Vec2.v);
		Vec1 = *(vec4*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x + Vec2.x), (Vec1.y + Vec2.y), (Vec1.z + Vec2.z), (Vec1.w + Vec2.w) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec4 operator+(vec4 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(Vec1.v, _mm_set1_ps(Value));
		return *(vec4*)&r;
#else
		vec4 Result = { (Vec1.x + Value), (Vec1.y + Value), (Vec1.z + Value), (Vec1.w + Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec4& operator+=(vec4& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_add_ps(Vec1.v, _mm_set1_ps(Value));
		Vec1 = *(vec4*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x + Value), (Vec1.y + Value), (Vec1.z + Value), (Vec1.w + Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec4 operator-(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(Vec1.v, Vec2.v);
		return *(vec4*)&r;
#else
		vec4 Result = { (Vec1.x - Vec2.x), (Vec1.y - Vec2.y), (Vec1.z - Vec2.z), (Vec1.w - Vec2.w) };
		return(Result);
#endif
	}

	SYL_INLINE vec4& operator-=(vec4& Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(Vec1.v, Vec2.v);
		Vec1 = *(vec4*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x - Vec2.x), (Vec1.y - Vec2.y), (Vec1.z - Vec2.z), (Vec1.w - Vec2.w) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec4 operator-(vec4 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(Vec1.v, _mm_set1_ps(Value));
		return *(vec4*)&r;
#else
		vec4 Result = { (Vec1.x - Value), (Vec1.y - Value), (Vec1.z - Value), (Vec1.w - Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec4& operator-=(vec4& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(Vec1.v, _mm_set1_ps(Value));
		Vec1 = *(vec4*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x - Value), (Vec1.y - Value), (Vec1.z - Value), (Vec1.w - Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec4 operator-(float Value, vec4 Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_set1_ps(Value), Vec1.v);
		return *(vec4*)&r;
#else
		vec4 Result = { (Value - Vec1.x), (Value - Vec1.y), (Value - Vec1.z), (Value - Vec1.w) };
		return(Result);
#endif
	}

	SYL_INLINE vec4& operator-=(float Value, vec4& Vec1)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_sub_ps(_mm_set1_ps(Value), Vec1.v);
		Vec1 = *(vec4*)&r;
		return(Vec1);
#else
		Vec1 = { (Value - Vec1.x), (Value - Vec1.y), (Value - Vec1.z), (Value - Vec1.w) };
		return(Vec1);
#endif
	}


	SYL_INLINE vec4 operator*(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(Vec1.v, _mm_load_ps(Vec2.e));
		return *(vec4*)&r;
#else
		vec4 Result = { (Vec1.x - Vec2.x), (Vec1.y - Vec2.y), (Vec1.z - Vec2.z), (Vec1.w - Vec2.w) };
		return(Result);
#endif
	}

	SYL_INLINE vec4& operator*=(vec4& Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(Vec1.v, _mm_load_ps(Vec2.e));
		Vec1 = *(vec4*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x * Vec2.x), (Vec1.y * Vec2.y), (Vec1.z * Vec2.z), (Vec1.w * Vec2.w) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec4 operator*(vec4 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(Vec1.v, _mm_set1_ps(Value));
		return *(vec4*)&r;
#else
		vec4 Result = { (Vec1.x * Value), (Vec1.y * Value), (Vec1.z * Value), (Vec1.w * Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec4& operator*=(vec4& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_mul_ps(Vec1.v, _mm_set1_ps(Value));
		Vec1 = *(vec4*)&r;
		return(Vec1);
#else
		Vec1 = { (Vec1.x * Value), (Vec1.y * Value), (Vec1.z * Value), (Vec1.w * Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec4 operator/(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 r = _mm_div_ps(Vec1.v, Vec2.v);
		return *(vec4*)&r;
#else
		vec4 Result = { (Vec1.x / Vec2.x), (Vec1.y / Vec2.y), (Vec1.z / Vec2.z), (Vec1.w / Vec2.w) };
		return(Result);
#endif
	}

	SYL_INLINE vec4& operator/=(vec4& Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_div_ps(Vec1.v, Vec2.v);
		Vec1 = *(vec4*)&Result;
		return(Vec1);
#else
		Vec1 = { (Vec1.x / Vec2.x), (Vec1.y / Vec2.y), (Vec1.z / Vec2.z), (Vec1.w / Vec2.w) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec4 operator/(vec4 Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_div_ps(Vec1.v, _mm_set1_ps(Value));
		return *(vec4*)&Result;
#else
		vec4 Result = { (Vec1.x / Value), (Vec1.y / Value), (Vec1.z / Value), (Vec1.w / Value) };
		return(Result);
#endif
	}

	SYL_INLINE vec4& operator/=(vec4& Vec1, float Value)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_div_ps(Vec1.v, _mm_set1_ps(Value));
		Vec1 = *(vec4*)&Result;
		return(Vec1);
#else
		Vec1 = { (Vec1.x / Value), (Vec1.y / Value), (Vec1.z / Value), (Vec1.w / Value) };
		return(Vec1);
#endif
	}

	SYL_INLINE vec4 Floor(vec4 A)
	{
		vec4 Result = { Floor(A.x), Floor(A.y), Floor(A.z), Floor(A.w) };
		return(Result);
	}

	/* Round all components of the vec3 to nearest integer*/
	SYL_INLINE vec4 Round(vec4 A)
	{
		vec4 Result = { Round(A.x), Round(A.y), Round(A.z), Round(A.w) };
		return(Result);
	}

	/* Negate all components of the vector */
	SYL_INLINE vec4 Negate(vec4 a)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_sub_ps(_mm_set1_ps(0), a.v);
		return *(vec4*)&Result;
#else
		vec4 Result = { -a.x, -a.y, -a.z, -a.w };
		return(Result);
#endif
	}

	SYL_INLINE float Dot(vec4 Vec1, vec4 Vec2)
	{
#if defined(SYL_DEBUG)
		return((Vec1.x * Vec2.x) + (Vec1.y * Vec2.y) + (Vec1.z * Vec2.z) + (Vec1.w * Vec2.w));
#else
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_dp_ps(Vec1.v, Vec2.v, 0xFF);
		return(*(float*)&Result);
#else
		return (Vec1.e[0] * Vec2.e[0]) + (Vec1.e[1] * Vec2.e[1]) + (Vec1.e[2] * Vec2.e[2]) + (Vec1.e[3] * Vec2.e[3]);
#endif
#endif
	}

	SYL_INLINE vec4 Hadamard(vec4 Vec1, vec4 Vec2)
	{
		vec4 Result = { (Vec1.x * Vec2.x), (Vec1.y * Vec2.y), (Vec1.z * Vec2.z), (Vec1.w * Vec2.w) };
		return(Result);
	}

	SYL_INLINE float Length(vec4 Vec1)
	{
#if defined(SYL_DEBUG)
		return(sqrt((Vec1.x * Vec1.x) + (Vec1.y * Vec1.y) + (Vec1.z * Vec1.z) + (Vec1.w * Vec1.w)));
#else
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 V = _mm_load_ps(Vec1.e);
		__m128 A1 = _mm_mul_ps(V, V);
		__m128 A2 = _mm_hadd_ps(A1, A1);
		__m128 A3 = _mm_hadd_ps(A2, A2);
		return sqrtf(_mm_cvtss_f32(A3));
#else
		return(sqrt((Vec1.x * Vec1.x) + (Vec1.y * Vec1.y) + (Vec1.z * Vec1.z) + (Vec1.w * Vec1.w)));
#endif
#endif
	}

	SYL_INLINE float Distance(vec4 Vec1, vec4 Vec2)
	{
		return Length(Vec1 - Vec2);
	}

	SYL_INLINE vec4 Normalize(vec4 a)
	{
		return(a * (1.0f / Length(a)));
	}

	SYL_INLINE vec4 Lerp(vec4 Vec1, vec4 Vec2, float t)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_add_ps(Vec1.v, _mm_mul_ps((_mm_sub_ps(Vec2.v, Vec1.v)), _mm_set1_ps(t)));
		return(*(vec4*)&Result);
#else
		vec4 Result = Vec1 + ((Vec2 - Vec1) * t);
		return(Result);
#endif
	}

	SYL_STATIC vec4 Cross(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 a1 = _SYL_PERMUTE_PS(Vec1.v, _SYL_SHUFFLE(3, 0, 2, 1));
		__m128 a2 = _SYL_PERMUTE_PS(Vec2.v, _SYL_SHUFFLE(3, 1, 0, 2));
		__m128 r = _mm_mul_ps(a1, a2);
		a1 = _SYL_PERMUTE_PS(a1, _SYL_SHUFFLE(3, 0, 2, 1));
		a2 = _SYL_PERMUTE_PS(a2, _SYL_SHUFFLE(3, 1, 0, 2));
		r = _SYL_ADD_PS(a1, a2, r);
		__m128 re = _mm_and_ps(r, _S_XMM_MASK_3);
		return *(vec4*)&re;
#else
		vec4 Result = { Vec1.e[1] * Vec2.e[2] - Vec1.e[2] * Vec2.e[1],
						 Vec1.e[2] * Vec2.e[0] - Vec1.e[0] * Vec2.e[2],
						 Vec1.e[0] * Vec2.e[1] - Vec1.e[1] * Vec2.e[0], 0.0f };
		return(Result);
#endif
	}

	SYL_INLINE vec4 Project(vec4 VectorToProject, vec4 ProjectionVector)
	{
		float scale = Dot(ProjectionVector, VectorToProject) / Dot(ProjectionVector, ProjectionVector);
		return(ProjectionVector * scale);
	}

	/* Per component comparsion to return a vector containing the largest components */
	SYL_INLINE vec4 MaxVector(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_max_ps(Vec1.v, Vec2.v);
		return *(vec4*)&Result;
#else
		vec4 Result;
		if (Vec1.x > Vec2.x)
		{
			Result.x = Vec1.x;
		}
		else
		{
			Result.x = Vec2.x;
		}

		if (Vec1.y > Vec2.y)
		{
			Result.y = Vec1.y;
		}
		else
		{
			Result.y = Vec2.y;
		}

		if (Vec1.z > Vec2.z)
		{
			Result.z = Vec1.z;
		}
		else
		{
			Result.z = Vec2.z;
		}

		if (Vec1.w > Vec2.w)
		{
			Result.w = Vec1.w;
		}
		else
		{
			Result.w = Vec2.w;
		}
		return(Result);
#endif
	}

	/* Per component comparsion to return a vector containing the smollest components */
	SYL_INLINE vec4 MinVector(vec4 Vec1, vec4 Vec2)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Result = _mm_min_ps(Vec1.v, Vec2.v);
		return *(vec4*)&Result;
#else
		vec4 Result;
		if (Vec1.x < Vec2.x)
		{
			Result.x = Vec1.x;
		}
		else
		{
			Result.x = Vec2.x;
		}

		if (Vec1.y < Vec2.y)
		{
			Result.y = Vec1.y;
		}
		else
		{
			Result.y = Vec2.y;
		}

		if (Vec1.z < Vec2.z)
		{
			Result.z = Vec1.z;
		}
		else
		{
			Result.z = Vec2.z;
		}

		if (Vec1.w < Vec2.w)
		{
			Result.w = Vec1.w;
		}
		else
		{
			Result.w = Vec2.w;
		}

		return(Result);
#endif
	}

	SYL_INLINE vec4 Clamp(vec4 Value, vec4 Min, vec4 Max)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		// __m128 Result = _mm_max_ps(Min.v, Max.v);
		// Result = _mm_max_ps(Max.v, Result);
		// return *(vec4*)&Result;

		__m128 Result = _mm_min_ps(_mm_max_ps(Value.v, Min.v), Max.v);
		return *(vec4*)&Result;
#else
		vec4 Result = MinVector(MaxVector(Value, Min), Max);
		return(Result);
#endif
	}

	/* Return the biggest element inside vec4 */
	SYL_INLINE float Max(vec4 A)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Value = _mm_load_ps(A.e);
		__m128 r = _mm_max_ps(Value, Value);
		return *(float*)&r;
#else
		/* TODO(xcatalyst): Implement ! */
		return(A.e[0]);
#endif
	}

	/* Return the smollest element inside vec4 */
	SYL_INLINE float Min(vec4 A)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Value = _mm_load_ps(A.e);
		__m128 r = _mm_min_ps(Value, Value);
		return *(float*)&r;
#else
		/* TODO(xcatalyst): Implement ! */
		return(A.e[0]);
#endif
	}

	SYL_INLINE float Sum(vec4 Vec1)
	{
		return((Vec1.x + Vec1.y) + (Vec1.z + Vec1.w));
	}

	/*********************************************
	*                 MATRIX 4X4                 *
	*********************************************/

	SYL_INLINE mat4 MAT4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		mat4 R;
		R.v[0] = _mm_set_ps(m00, m01, m02, m03);
		R.v[1] = _mm_set_ps(m10, m11, m12, m13);
		R.v[2] = _mm_set_ps(m20, m21, m22, m23);
		R.v[3] = _mm_set_ps(m30, m31, m32, m33);
		return(R);
#else
		mat4 Result = { m00, m01, m02, m03,
			  m10, m11, m12, m13,
			  m20, m21, m22, m23,
			  m30, m31, m32, m33 };
		return(Result);
#endif
	}

	SYL_INLINE mat4 MAT4(float* a)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		mat4 R;
		R.v[0] = _mm_set_ps(a[0], a[1], a[2], a[3]);
		R.v[1] = _mm_set_ps(a[4], a[5], a[6], a[7]);
		R.v[2] = _mm_set_ps(a[8], a[9], a[10], a[11]);
		R.v[3] = _mm_set_ps(a[12], a[13], a[14], a[15]);
		return(R);
#else
		mat4 Result = { a[0], a[1], a[2], a[3],
				   a[4], a[5], a[6], a[7],
				   a[8], a[9], a[10], a[11],
				   a[12], a[13], a[14], a[15] };
		return(Result);
#endif
	}

	SYL_INLINE void Zero(mat4& Matrix)
	{
#if defined(SYL_ENABLE_AVX)
		_mm256_store_ps(Matrix.e, _S_YMM_ZERO);
		_mm256_store_ps(Matrix.e + 8, _S_YMM_ZERO);
#elif defined(SYL_ENABLE_SSE4)
		_mm_store_ps(Matrix.e, _S_XMM_ZERO);
		_mm_store_ps(Matrix.e + 4, _S_XMM_ZERO);
		_mm_store_ps(Matrix.e + 8, _S_XMM_ZERO);
		_mm_store_ps(Matrix.e + 12, _S_XMM_ZERO);
#else
		Matrix = _S_ZERO4X4;
#endif
	}

	SYL_INLINE mat4 Identity()
	{
		return(_S_IDENT4X4);
	}

	SYL_STATIC bool IsIdentity(mat4 Mat)
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
			Mat.e[12] == 0.0f && Mat.e[13] == 0.0f && Mat.e[14] == 0.0f && Mat.e[15] == 1.0f)
		{
			/* Nice you spend couple of frames calculating this... */
			return(true);
		}
		else
		{
			/* Maybe even failed what a shame.. */
			return(false);
		}
#endif
	}

	/* Multiply two 4x4 Matricies */
	SYL_STATIC mat4 operator*(mat4 Matrix1, mat4 Matrix2)
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

		mat4 Result = {};

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
		mat4 Result;
		Zero(Result);

		for (int k = 0; k < 4; ++k)
		{
			for (int n = 0; n < 4; ++n)
			{
				for (int i = 0; i < 4; ++i)
				{
					Result.e2[k][n] += Matrix1.e2[k][i] * Matrix2.e2[i][n];
				}
			}
		}

		return(Result);
#endif
	}

	SYL_STATIC mat4 Transpose(mat4 Mat)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 Temp1 = _mm_shuffle_ps(Mat.v[0], Mat.v[1], _SYL_SHUFFLE(1, 0, 1, 0));
		__m128 Temp2 = _mm_shuffle_ps(Mat.v[0], Mat.v[1], _SYL_SHUFFLE(3, 2, 3, 2));
		__m128 Temp3 = _mm_shuffle_ps(Mat.v[2], Mat.v[3], _SYL_SHUFFLE(1, 0, 1, 0));
		__m128 Temp4 = _mm_shuffle_ps(Mat.v[2], Mat.v[3], _SYL_SHUFFLE(3, 2, 3, 2));
		mat4 R;
		R.v[0] = _mm_shuffle_ps(Temp1, Temp2, _SYL_SHUFFLE(2, 0, 2, 0));
		R.v[1] = _mm_shuffle_ps(Temp1, Temp2, _SYL_SHUFFLE(3, 1, 3, 1));
		R.v[2] = _mm_shuffle_ps(Temp3, Temp4, _SYL_SHUFFLE(2, 0, 2, 0));
		R.v[3] = _mm_shuffle_ps(Temp3, Temp4, _SYL_SHUFFLE(3, 1, 3, 1));
		return(R);
#else
		mat4 Result;

		for (int j = 0; j < 4; ++j)
		{
			for (int i = 0; i < 4; ++i)
			{
				Result.e2[j][i] = Mat.e2[i][j];
			}
		}
		return(Result);
#endif
	}

	/* Inverse of a matrix but in the scale of this matrix should be 1 */
	SYL_STATIC mat4 InverseNoScale(mat4 Matrix)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)

		__m128 Temp0 = _SYL_VEC_SHUFFLE_0101(Matrix.v[0], Matrix.v[1]); // 00, 01, 10, 11
		__m128 Temp1 = _SYL_VEC_SHUFFLE_2323(Matrix.v[0], Matrix.v[1]); // 02, 03, 12, 13

		mat4 Result;
		Result.v[0] = _SYL_VEC_SHUFFLE(Temp0, Matrix.v[2], 0, 2, 0, 3); // 00, 10, 20, 23(=0)
		Result.v[1] = _SYL_VEC_SHUFFLE(Temp0, Matrix.v[2], 1, 3, 1, 3); // 01, 11, 21, 23(=0)
		Result.v[2] = _SYL_VEC_SHUFFLE(Temp1, Matrix.v[2], 0, 2, 2, 3); // 02, 12, 22, 23(=0)

		Result.v[3] = _mm_mul_ps(Result.v[0], _SYL_VEC_SWIZZLE1(Matrix.v[3], 0));
		Result.v[3] = _mm_add_ps(Result.v[3], _mm_mul_ps(Result.v[1], _SYL_VEC_SWIZZLE1(Matrix.v[3], 1)));
		Result.v[3] = _mm_add_ps(Result.v[3], _mm_mul_ps(Result.v[2], _SYL_VEC_SWIZZLE1(Matrix.v[3], 2)));
		Result.v[3] = _mm_sub_ps(_mm_setr_ps(0.f, 0.f, 0.f, 1.f), Result.v[3]);
		return(Result);
#else
		mat4 x;
		assert(1 == 0 && "InverseNoScale() non-sse version not implemented!");
		return(x);
#endif
	}

	/* Inverse of a matrix*/
	SYL_STATIC mat4 Inverse(mat4 Matrix)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		__m128 a0 = _SYL_VEC_SHUFFLE_0101(Matrix.v[0], Matrix.v[1]); // 00, 01, 10, 11
		__m128 a1 = _SYL_VEC_SHUFFLE_2323(Matrix.v[0], Matrix.v[1]); // 02, 03, 12, 13

		mat4 Result;
		Result.v[0] = _SYL_VEC_SHUFFLE(a0, Matrix.v[2], 0, 2, 0, 3); // 00, 10, 20, 23(=0)
		Result.v[1] = _SYL_VEC_SHUFFLE(a0, Matrix.v[2], 1, 3, 1, 3); // 01, 11, 21, 23(=0)
		Result.v[2] = _SYL_VEC_SHUFFLE(a1, Matrix.v[2], 0, 2, 2, 3); // 02, 12, 22, 23(=0)

		__m128 SizeSquared = _mm_mul_ps(Result.v[0], Result.v[0]);
		SizeSquared = _mm_add_ps(SizeSquared, _mm_mul_ps(Result.v[1], Result.v[1]));
		SizeSquared = _mm_add_ps(SizeSquared, _mm_mul_ps(Result.v[2], Result.v[2]));

		__m128 Sqr = _mm_blendv_ps(_mm_div_ps(_S_XMM_ZERO, SizeSquared), _S_XMM_ZERO, _mm_cmplt_ps(SizeSquared, _mm_set1_ps(_SYL_SMALL_NUMBER)));

		Result.v[0] = _mm_mul_ps(Result.v[0], Sqr);
		Result.v[1] = _mm_mul_ps(Result.v[1], Sqr);
		Result.v[2] = _mm_mul_ps(Result.v[2], Sqr);

		Result.v[3] = _mm_mul_ps(Result.v[0], _SYL_VEC_SWIZZLE1(Matrix.v[3], 0));
		Result.v[3] = _mm_add_ps(Result.v[3], _mm_mul_ps(Result.v[1], _SYL_VEC_SWIZZLE1(Matrix.v[3], 1)));
		Result.v[3] = _mm_add_ps(Result.v[3], _mm_mul_ps(Result.v[2], _SYL_VEC_SWIZZLE1(Matrix.v[3], 2)));
		Result.v[3] = _mm_sub_ps(_mm_setr_ps(0.f, 0.f, 0.f, 1.f), Result.v[3]);

		return(Result);
#else
		mat4 x;
		assert(1 == 0 && "Inverse() non-sse version not implemented!");
		return(x);
#endif
	}

	SYL_STATIC vec4 Transform(mat4 Matrix, vec4 Vector)
	{
#if defined (SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		vec4 Result;
		Result.v = _SYL_PERMUTE_PS(Vector.v, _MM_SHUFFLE(3, 3, 3, 3));
		Result.v = _mm_mul_ps(Result.v, Matrix.v[3]);
		__m128 Temp = _SYL_PERMUTE_PS(Vector.v, _MM_SHUFFLE(2, 2, 2, 2));
		Result.v = _SYL_ADD_PS(Temp, Matrix.v[2], Result.v);
		Temp = _SYL_PERMUTE_PS(Vector.v, _MM_SHUFFLE(1, 1, 1, 1));
		Result.v = _SYL_ADD_PS(Temp, Matrix.v[1], Result.v);
		Temp = _SYL_PERMUTE_PS(Vector.v, _MM_SHUFFLE(0, 0, 0, 0));
		Result.v = _SYL_ADD_PS(Temp, Matrix.v[0], Result.v);
		return(Result);
#else
		vec4 Result;

		Result.x = Vector.x * Matrix.e2[0][0] + Vector.y * Matrix.e2[0][1] + Vector.z * Matrix.e2[0][2] + Vector.w * Matrix.e2[0][3];
		Result.y = Vector.x * Matrix.e2[1][0] + Vector.y * Matrix.e2[1][1] + Vector.z * Matrix.e2[1][2] + Vector.w * Matrix.e2[1][3];
		Result.z = Vector.x * Matrix.e2[2][0] + Vector.y * Matrix.e2[2][1] + Vector.z * Matrix.e2[2][2] + Vector.w * Matrix.e2[2][3];
		Result.w = Vector.x * Matrix.e2[3][0] + Vector.y * Matrix.e2[3][1] + Vector.z * Matrix.e2[3][2] + Vector.w * Matrix.e2[3][3];

		return(Result);
#endif
	}

	SYL_INLINE vec4 operator*(mat4 Matrix1, vec4 Vector)
	{
		return Transform(Matrix1, Vector);
	}

	SYL_INLINE vec3 operator*(mat4 Matrix1, vec3 Vector)
	{
		vec4 Vec = Transform(Matrix1, VEC4(Vector, 1.0f));
		vec3 Result = { Vec.x, Vec.y, Vec.z };
		return(Result);
	}

	SYL_STATIC mat4 XRotation(float Angle)
	{
		float CosAngle = cos(Angle);
		float SinAngle = sin(Angle);

		mat4 Result = {
			1, 0, 0, 0,
			0, CosAngle,-SinAngle, 0,
			0, SinAngle, CosAngle, 0,
			0, 0, 0, 1
		};

		return(Result);
	}

	SYL_STATIC mat4 YRotation(float Angle)
	{
		float CosAngle = cos(Angle);
		float SinAngle = sin(Angle);

		mat4 Result = {
			CosAngle, 0, SinAngle, 0,
			0, 1, 0, 0,
			-SinAngle, 0, CosAngle, 0,
			0, 0, 0, 1
		};

		return(Result);
	}

	SYL_STATIC mat4 ZRotation(float Angle)
	{
		float CosAngle = cos(Angle);
		float SinAngle = sin(Angle);

		mat4 Result = {
			CosAngle, -SinAngle, 0, 0,
			SinAngle, CosAngle, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1
		};

		return(Result);
	}

	SYL_STATIC mat4 Translation(vec3 Vector)
	{
#if defined(SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		mat4 Result;
		Result.v[0] = _S_IDENT4x4R0;
		Result.v[1] = _S_IDENT4x4R1;
		Result.v[2] = _S_IDENT4x4R2;
		Result.v[0] = _mm_set_ps(Vector.x, Vector.y, Vector.z, 1.0f);
		return(Result);
#else
		mat4 Result =
		{
				1, 0, 0, Vector.x,
				0, 1, 0, Vector.y,
				0, 0, 1, Vector.z,
				0, 0, 0, 1,
		};

		return(Result);
#endif
	}

	/* No LH version for now...  */
	SYL_STATIC mat4 PerspectiveProjectionRH(float Fov, float AspectRatio, float NearClipPlane, float FarClipPlane)
	{
#if defined(SYL_ENABLE_SSE4) || defined(SYL_ENABLE_AVX)
		float Sin = sin(0.5f * Fov);
		float Cos = cos(0.5f * Fov);
		float Range = FarClipPlane / (NearClipPlane - FarClipPlane);
		float Height = Cos / Sin;
		__m128 Mem = { Height / AspectRatio, Height, Range, Range * NearClipPlane };
		__m128 Mem2 = Mem;
		__m128 Temp = _mm_setzero_ps();
		Temp = _mm_move_ss(Temp, Mem2);
		mat4 Result;
		Result.v[0] = Temp;
		Temp = Mem2;
		Temp = _mm_and_ps(Temp, _S_XMM_MASK_Y);
		Result.v[1] = Temp;
		Temp = _mm_setzero_ps();
		Mem2 = _mm_shuffle_ps(Mem2, _S_IDENT4x4R3, _MM_SHUFFLE(3, 2, 3, 2));
		Temp = _mm_shuffle_ps(Temp, Mem2, _MM_SHUFFLE(3, 0, 0, 0));
		Result.v[2] = Temp;
		Temp = _mm_shuffle_ps(Temp, Mem2, _MM_SHUFFLE(2, 1, 0, 0));
		Result.v[3] = Temp;
		return(Result);
#else
		//float Cotan = 1.0f / tanf(Fov / 2.0f);

		//mat4 Result = { 
		//	Cotan / AspectRatio, 0.0f, 0.0f, 0.0f,
		//	0.0f, Cotan, 0.0f, 0.0f,
		//	0.0f, 0.0f, (FarClipPlane + NearClipPlane) / (NearClipPlane - FarClipPlane), -1.0f,
		//	0.0f, 0.0f, (2.0f * FarClipPlane * NearClipPlane) / (NearClipPlane - FarClipPlane), 0.0f 
		//};

		//return(Result);

		float tanHalfFovy = tan(Fov / 2);

		mat4 Result;
		Zero(Result);
		Result.e2[0][0] = 1 / (AspectRatio * tanHalfFovy);
		Result.e2[1][1] = 1 / (tanHalfFovy);
		Result.e2[2][2] = -(FarClipPlane + NearClipPlane) / (FarClipPlane - NearClipPlane);
		Result.e2[2][3] = -1;
		Result.e2[3][2] = -(2 * FarClipPlane * NearClipPlane) / (FarClipPlane - NearClipPlane);
		return Result;
#endif
	}

	SYL_STATIC mat4 OrthographicProjectionRH(float AspectRatio, float NearClipPlane, float FarClipPlane)
	{
		float Ral = 1.0f;
		float Rsl = AspectRatio;
		float Fan = NearClipPlane;
		float Fsn = FarClipPlane;
		float Tab = 2.0f / (Fan - Fsn);
		float Tsb = (Fan + Fsn) / (Fan - Fsn);

		mat4 Result =
		{ 1 / Ral,   0,   0,    0,
		 0, 1 / Rsl,   0,    0,
		 0,   0, 1 / Tab, -Tsb / Tab,
		 0,   0,   0,    1 };
		return(Result);
	}

}

#endif // SYLVESTER_H
