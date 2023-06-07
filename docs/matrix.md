# Matrix Functions for C

* Sylvester use column major matricies.

Set all values of a matrix to the value
```cpp
smat4 SMAT4(float value);
```

Returns a identity matrix.
```cpp
mat4 s_mat4_identity();
```

Checks if a matrix is identity matrix.
```cpp
bool s_mat4_identity(mat4 Mat);
```

Return transpose of a matrix
```cpp
mat4 s_mat4_transpose(mat4 Mat);
```

Translate matrix with the given vector
```cpp
smat4 s_mat4_translate(smat4 matrix, svec3 vec);
```

Rotate the matrix with given angle and vector
```cpp
smat4 s_mat4_rotate(smat4 matrix, float angle, svec3 vec)
```

Return inverse of a matrix.
```cpp
mat4 s_mat4_inverse(mat4 Matrix);
```

Inverse of a matrix but the scale of this matrix should be 1
```cpp
mat4 s_mat4_inverse_noscale(mat4 Matrix);
```

Transform the vector by the given matrix
```cpp
vec4 s_mat4_transform(mat4 Matrix, vec4 Vector)
```

Rotate the matrix along X axis.
```cpp
mat4 s_mat4_xrotation(float Angle);
```

Rotate the matrix along Y axis.
```cpp
mat4 s_mat4_yrotation(float Angle);
```

Rotate the matrix along Z axis.
```cpp
mat4 s_mat4_zrotation(float Angle);
```

Build a translation matrix from a 3d vector.
```cpp
mat4 s_mat4_translation(vec3 Vector);
```

Return a projection matrix in RH with the given values.
```cpp
mat4 s_mat4_perspective_projection_rh(float Fov, float AspectRatio, float NearClipPlane, float FarClipPlane);
```

Return a ortographic matrix with the given values.
```cpp
mat4 s_mat4_orthographic_projection_rh(float AspectRatio, float NearClipPlane, float FarClipPlane);
`
