# Matrix functions for CPP

* Sylvester use column major matricies.

Returns a identity matrix.
```cpp
mat4 Identity();
```

Checks if a matrix is identity matrix.
```cpp
bool IsIdentity(mat4 Mat);
```

Return transpose of a matrix
```cpp
mat4 Transpose(mat4 Mat);
```

Return inverse of a matrix.
```cpp
mat4 Inverse(mat4 Matrix);
```

Inverse of a matrix but the scale of this matrix should be 1 
```cpp
InverseNoScale(mat4 Matrix);
```

Transform the vector by the given matrix
```cpp
vec4 Transform(mat4 Matrix, vec4 Vector)
```

Rotate the matrix along X axis.
```cpp
mat4 XRotation(float Angle);
```

Rotate the matrix along Y axis.
```cpp
mat4 YRotation(float Angle);
```

Rotate the matrix along Z axis.
```cpp
mat4 ZRotation(float Angle);
```

Build a translation matrix from a 3d vector.
```cpp
mat4 Translation(vec3 Vector);
```

Return a projection matrix in RH with the given values.
```cpp
mat4 PerspectiveProjectionRH(float Fov, float AspectRatio, float NearClipPlane, float FarClipPlane);
```

Return a ortographic matrix with the given values. 
```cpp
mat4 OrthographicProjection(float AspectRatio, float NearClipPlane, float FarClipPlane);
```