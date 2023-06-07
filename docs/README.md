Sylvester Introduction
====

Copy the [sylvester.h](https://github.com/xcatalyst/Sylvester/blob/master/sylvester.h) into your project and include it.
Sylvester is a stb style header only library so if you want the definitions you can ```#define SYL_IMPLEMENTATION``` otherwise
only decleartions will be included. Before including the library  you
can enable couple of features or disable them. Sylvester supports SSE4 and AVX. If you want to enable them in your
project decleare their flags before including the library like so:
```cpp
#define SYL_ENABLE_SSE4 // Or SYL_ENABLE_AVX for AVX support
#include "sylvester.h"
```
There is also ```SYL_DEBUG``` flag which changes the implementations of some of the(not much)
functions in the library to versions which work faster on debug builds. Note that
this is benchmarked on MSVC and for other compilers ```SYL_DEBUG``` flag might not effect much.
This flag is only for having higher performance on debug builds, it's a bit overkill.

Sylvester does not use intrinsic by default. If your architecture supports AVX you
can enable them by adding ```#define SYL_ENABLE_AVX```, for SSE ```#define SYL_ENABLE_SSE4```
Sylvester currently doesn't support NEON.

If you want to make function non-inline (NOT RECCOMENDED) define ```SYL_NO_INLINE``` \
If you want to mark the functions as non static define ```SYL_NO_STATIC``` \
Before including ```sylvester.h```

[C version guide](#c-introduction) \

# Introduction
All Sylvester function are prefixed with ```s_```. All of the operator comparisons have
to go trough a inline function, named very straightforward in the following style:

|   s_   | vec2 |    add    |                scalar            |
|--------|------|-----------|----------------------------------|
| prefix | type | operation | type to do the operation against |

In the above the function is written as ```s_vec2_add_scalar``` all with underscore. This function
adds scalar value to the given vec2 and returns. If the type has ```p``` at the and i.e. ```vec2p```
this adds the scalar value to that variable (Pass by referance) and returns a vec2 result containing
the final vector. You don't have to assign the return type to anything. It's only for preferance.

----

### Utility Functions

There aren't many functions in this section. I currently don't have
any idea for what I can add. Reccomendations are appreciated.
[List of utility functions for C](https://github.com/xcatalyst/Sylvester/blob/master/docs/utility_c.md)

----

### Vector Operations

[List of vector functions for C](https://github.com/xcatalyst/Sylvester/blob/master/docs/vector_c.md)

There are 3 types of vectors supported on Sylvester. 4D, 3D and 2D vectors which can be
accessed with ```svec2``` ```svec3``` and ```svec4```.

```cpp
S_VEC3(float a, float b, float c); // Takes floats
S_VEC3A(float* a);                  // Takes array
```
I am not going to be typing out every possible vector function out there. Because the only thing that
changes are prefixes.

Comparing between two types can be done with functions. Not that these operators have orders and
they all have the same operation order.
```cpp
s_vec2_mul_scalar(Vector * Scalar);
s_vec2_mul(Vector * Vector); // Doesn't take another prefix at the end if both types are the same as the first one
s_mat4_mul_vec4(Matrix * Vector);
```

You cannot compare ```s_vec2_equal_scalar(4.0f, Vector)```, Vector comes first ```s_vec2_equal_scalar(Vector, 4.0f)```. Same thing for
matricies as well. You cannot multiply ```s_mat4_mul_vec4(Vector, Matrix)```, the correct way of doing it
would be ```s_mat4_mul_vec4(Matrix, Vector)```. The parameters are in the same order as in the function names.

### Matrix Operations

[List of matrix functions](https://github.com/xcatalyst/Sylvester/blob/master/docs/matrix_c.md)

Sylvester matrix is decleared as smat4. Currently there is only 4x4 matrix in Sylvester. I might
add 3x3 later but currently priority is given to make the library more robust then to add
new features. mat4 accepts multiple ways to acceces parameters.
```cpp
mat4 : { float m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 } // Individual
	{ float e[16] }    // Array
        { float e2[4][4] } // 2D access
        { vec4 v4d[4] }    // Access it with 4d vectors
        { __m128 v[4] }    // If defined SYL_ENABLE_SSE4
        { __m256 v2[2] }   // If defined SYL_ENABLE_AVX
```

mat4 can be zeroed out with intrinsics with the function ```s_mat4_zero(mat4 *Matrix)```. This function also zeros out vectors but I only implemented
it so it wouldn't be only for a single type. There is no need to use ```s_vector_type_zero(vector_type Vector)``` for vectors. The compiler should be able to optimize it with sse loads. You can define a matrix with the following functions:
```cpp
MAT4F(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33); // With values
MAT4A(float *a); // With array
MAT4V(vec4 a, vec4 b, vec4 c, vec4 d); // Multiple vectors
```

----

 ### To Be added
 Quaternion\
 Euler Angles\
 Noise Generation\
 Color functions
