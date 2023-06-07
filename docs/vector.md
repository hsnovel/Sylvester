# Vector Functions for C

Almost all of the vector functions are the same. The only thing that change
is their type. For not being too spesific instead of typing out the types in
the function I will use ```vector_type``` type which you can replace with
the type of vector you want. It's only for  demonstration, there isn't a 
```vector_type``` type in Sylvester.

This applies to function names as well. ```vector_type s_vector_type_negate(vector_type a)```
for negating a 2d vector would be ```vector_type s_vec2_negate(svec2 a)```

Note that the comparsion is done with AND. If ALL VectorB's corresponding elements
to VectorA not bigger than the comparison the operation will return false.

Only scalar versions have ```s_scalar_sub_vec2``` type of functions where
the scalar part is subtracted from all vector's elements, they are only for
vectors and only apply to subtract and divide as for the other operators the
result would stay the same.

---

### Functions

Negate all components of a vector.
```cpp
vector_type s_vector_type_negate(vector_type a);
```

Floor all components of a vector.
```cpp
vector_type s_vector_type_floor(vector_type A);
```

Round all components of a vector to the nearest integer.
```cpp
vector_type s_vector_type_round(vector_type A);
```

Dot product
```cpp
float s_vector_type_dot(vector_type Vec1, vector_type Vec2);
```

Hadamard product.
```cpp
vector_type s_vector_type_hadamard(vector_type Vec1, vector_type Vec2);
```

Linear interpolation. Construct new data points within the range of a given points. [For more information](https://en.wikipedia.org/wiki/Linear_interpolation)
```cpp
vector_type s_vector_type_lerp(vector_type Vec1, vector_type Vec2, float t);
```

Clamp all values of a vector to min and max values.
```cpp
vector_type s_vector_type_clamp(vector_type Value, vector_type Min, vector_type Max);
```

Length of the vector.
```cpp
float s_vector_type_length(vector_type Vec1);
```

Distance between two vectors.
```cpp
float s_vector_type_distance(vector_type Vec1, vector_type Vec2);
```

Normalize the given vector.
```cpp
vector_type s_vector_type_normalize(vector_type a)
```

Reflect a position to a normal plane. Only for 2d vectors.
```cpp
vector_type s_vector_type_reflect(vector_type Pos, vector_type N);
```

Project from a position along a vector on to a plain.
```cpp
vector_type s_vector_type_project(vector_type VectorToProject, vector_type ProjectionVector);
```

Flatten a position to a normal plane. Only for 2d vectors.
```cpp
vector_type s_vector_type_flatten(vector_type Pos, vector_type Normal);
```

Per component comparsion to return a vector containing the largest components.
```cpp
vector_type s_vector_type_max_vector(vector_type Vec1, vector_type Vec2);
```

Per component comparsion to return a vector containing the smollest components
```cpp
vector_type s_vector_type_max_vector_min_vector(vector_type Vec1, vector_type Vec2);
```

Return the biggest element.
```cpp
float s_vector_type_max(vector_type A);
```

Return the smollest element.
```cpp
float s_vector_type_min(vector_type A);
```

Add all components of the vector.
```cpp
float s_vector_type_sumum(vector_type Vec1);
```

Calculate the area of a triangle. Only for 2d vectors.
```cpp
float s_triangle_area(vec2 Vec1, vec2 Vec2, vec2 Vec3); // Only calculates 2D triangles
```

Cross product of two vectors. Only for 3d and 4d vectors.
```cpp
vector_type s_vector_type_cross(vector_type Vec1, vector_type Vec2)
```
