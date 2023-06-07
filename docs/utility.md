# Utility Functions for CPP

Convert radian to degree.
```cpp
float s_radian_to_degree(float Radian);
```

Convert degree to radian
```cpp
float s_degree_to_radian(float Degree);
```

Cast to int.
```cpp
float s_floorf(float A);
```

Cast to int.
```cpp
double s_floord(double A);
```

Round the float value to the nearest integer value.
```cpp
float s_roundf(float A);
```

Round the float value to the nearest integer value.
```cpp
double s_roundd(double A);
```

Round it to the nearest integer greater then given value.
```cpp
float s_ceilf(float A);
```

Round it to the nearest integer greater then given value.
```cpp
double s_ceild(double A);
```

Exctract BGRA color value into 4d vector.
```cpp
vec4 s_bgra_unpack(int Color);
```

Pack 4d vector into BGRA format color.
```cpp
unsigned int s_bgra_pack(vec4 Color);
```

Exctract RGBA color value into 4d vector.
```cpp
svec4 s_rgba_unpack(unsigned int Color);
```

Pack 4d vector into RGBA format color.
```cpp
unsigned int s_rgba_pack(svec4 Color);
```

Clamp value between minimum and maximum values.
```cpp
float s_clampf(float Value, float Min, float Max);
```

Clamp value between minimum and maximum values.
```cpp
double s_clampd(double Value, double Min, double Max);
```

Clamp value between minimum and maximum values.
```cpp
int s_clampi(int Value, int Min, int Max);
```

Clamp the value between 0 and 1.
```cpp
float s_clamp01f(float Value);
```

Clamp the value between 0 and 1.
```cpp
double s_clamp01d(double Value);
```

Clamp the value if it is above zero.
```cpp
float s_clamp_above_zero(float Value);
```

Clamp the value if it is below zero.
```cpp
float s_clamp_below_zero(float Value);
```

Returns true if the value is in the range of minumum and maximum. Otherwise returns false.
```cpp
bool s_is_in_range(float Value, float Min, float Max);
```
Linear interpolation. Construct new data points within the range of a given points. [For more information](https://en.wikipedia.org/wiki/Linear_interpolation)
```cpp
float s_lerp(float A, float t, float B);
```

Calculate the square. It is handy to type square if the expression happens to be very long.
```cpp
float s_square(float x);
```

Calculate the absoulte value.
```cpp
float s_abs(float x);
```

Find the hypotenuse of a triangle given two other sides
```cpp
float s_pythagorean(float x, float y);
```

Return the maximum value.
```cpp
float s_max(float x, float y);
```

Return the minimum value.
```cpp
float s_min(float x, float y);
```

Calculate the floating point remainder x divided by y.
```cpp
float s_mod(float x, float y);
```

Power raised to the base number
```cpp
float s_pow(float Base, float Power);
```

Return truncated value, with the ```Remain``` amount of digits not truncated starting from left. 
```cpp
float s_truncate(float Value, float Remain);
```

Normalize a value.
```cpp
float s_normalize(float Value, float Min, float Max);
```

Map a value from one range to another range. 
```cpp
float s_map(float Value, float SourceMin, float SourceMax, float DestMin, float DestMax);
```

Convert given RGB value to HSV. HSV values are mapped to X, Y and Z values of a vector in order.
```cpp
vec3 s_rgb_to_hsv(vec3 RGB);
```
