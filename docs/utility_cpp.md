# Utility Functions for C

Convert radian to degree.
```cpp
SYL_INLINE float RadianToDegree(float Radian);
```

Convert degree to radian
```cpp
SYL_INLINE float DegreeToRadian(float Degree);
```

Cast to int.
```cpp
SYL_INLINE float Floor(float A);
```

Round the float value to the nearest integer value.
```cpp
SYL_INLINE float Round(float A);
```

Round it to the nearest integer greater then given value.
```cpp
float Ceil(float A);
```

Exctract BGRA color value into 4d vector.
```cpp
SYL_INLINE vec4 BGRAUnpack(int Color);
```

Pack 4d vector into BGRA format color.
```cpp
SYL_INLINE unsigned int BGRAPack(vec4 Color);
```

Exctract RGBA color value into 4d vector.
```cpp
SYL_INLINE vec4 RGBAUnpack(unsigned int Color);
```

Pack 4d vector into RGBA format color.
```cpp
SYL_INLINE unsigned int RGBAPack(vec4 Color);
```

Clamp value between minimum and maximum values.
```cpp
SYL_INLINE float Clamp(float Value, float Min, float Max);
```

Clamp the value between 0 and 1.
```cpp
SYL_INLINE float Clamp01(float Value);
```

Clamp the value if it is above zero.
```cpp
SYL_INLINE float ClampAboveZero(float Value);
```

Clamp the value if it is below zero.
```cpp
SYL_INLINE float ClampBelowZero(float Value);
```

Returns true if the value is in the range of minumum and maximum. Otherwise returns false.
```cpp
SYL_INLINE bool IsInRange(float Value, float Min, float Max);
```
Linear interpolation. Construct new data points within the range of a given points. [For more information](https://en.wikipedia.org/wiki/Linear_interpolation)
```cpp
SYL_INLINE float Lerp(float A, float t, float B);
```

Calculate the square. It is handy to type square if the expression happens to be very long.
```cpp
float Square(float x);
```

Calculate the absoulte value.
```cpp
float Abs(float x);
```

Find the hypotenuse of a triangle given two other sides
```cpp
float Pythagorean(float x, float y);
```

Return the maximum value.
```cpp
float Max(float x, float y);
```

Return the minimum value.
```cpp
float Min(float x, float y);
```

Calculate the floating point remainder x divided by y.
```cpp
float Mod(float x, float y);
```

Power raised to the base number
```cpp
float Pow(float Base, float Power);
```

Return truncated value, with the ```Remain``` amount of digits not truncated starting from left. 
```cpp
float Truncate(float Value, float Remain);
```

Normalize a value.
```cpp
float Normalize(float Value, float Min, float Max);
```

Map a value from one range to another range. 
```cpp
float Map(float Value, float SourceMin, float SourceMax, float DestMin, float DestMax);
```

Convert given RGB value to HSV. HSV values are mapped to X, Y and Z values of a vector in order.
```cpp
vec3 RGBToHSV(vec3 RGB);
```
