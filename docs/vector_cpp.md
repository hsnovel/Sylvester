# Vector Functions for CPP

Almost all of the vector functions are the same. The only thing that change
is their type. For not being too spesific instead of typing out the types in
the function I will use ```vector_type``` type which you can replace with
the type of vector you want. It's only for  demonstration, there isn't a 
```vector_type``` type in Sylvester.

There is also operator overloading. You can simply compare multiply add etc..
with operator without the need of functions. A simple example would be:
```cpp
vec2 VectorA = { 1.0f, 2.0f };
vec2 VectorB = { 5.0f, 8.0f };

if(VectorB > VectorA)
  // Too easy huh ?
else
  // ...
```

Note that the comparsion is done with AND. If ALL VectorB's corresponding elements
to VectorA not bigger than the comparison the operation will return false.

---

### Functions

Negate all components of a vector.
```cpp
vector_type Negate(vector_type a);
```

Floor all components of a vector.
```cpp
vector_type Floor(vector_type A);
```

Round all components of a vector to the nearest integer.
```cpp
vector_type Round(vector_type A);
```

Dot product
```cpp
float Dot(vector_type Vec1, vector_type Vec2);
```

Hadamard product.
```cpp
vector_type Hadamard(vector_type Vec1, vector_type Vec2);
```

Linear interpolation. Construct new data points within the range of a given points. [For more information](https://en.wikipedia.org/wiki/Linear_interpolation)
```cpp
vector_type Lerp(vector_type Vec1, vector_type Vec2, float t);
```

Clamp all values of a vector to min and max values.
```cpp
vector_type Clamp(vector_type Value, vector_type Min, vector_type Max);
```

Length of the vector.
```cpp
float Length(vector_type Vec1);
```

Distance between two vectors.
```cpp
float Distance(vector_type Vec1, vector_type Vec2);
```

Normalize the given vector.
```cpp
vector_type Normalize(vector_type a)
```

Reflect a position to a normal plane. Only for 2d vectors.
```cpp
vector_type Reflect(vector_type Pos, vector_type N);
```

Project from a position along a vector on to a plain.
```cpp
vector_type Project(vector_type VectorToProject, vector_type ProjectionVector);
```

Flatten a position to a normal plane. Only for 2d vectors.
```cpp
vector_type Flatten(vector_type Pos, vector_type Normal);
```

Per component comparsion to return a vector containing the largest components.
```cpp
vector_type MaxVector(vector_type Vec1, vector_type Vec2);
```

Per component comparsion to return a vector containing the smollest components
```cpp
vector_type MinVector(vector_type Vec1, vector_type Vec2);
```

Return the biggest element.
```cpp
float Max(vector_type A);
```

Return the smollest element.
```cpp
float Min(vector_type A);
```

Add all components of the vector.
```cpp
float Sum(vector_type Vec1);
```

Calculate the area of a triangle. Only for 2d vectors.
```cpp
float AreaOfTriangle(vec2 Vec1, vec2 Vec2, vec2 Vec3);
```

Cross product of two vectors. Only for 3d and 4d vectors.
```cpp
vector_type Cross(vector_type Vec1, vector_type Vec2)
```
