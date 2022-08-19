# Dual Quaternion Benchmarks
This repository benchmarks a variety of dual quaternion product (dual quaternion 
multiplied by dual quaternion) implementations.

This project does not provide full-fledged quaternion or dual quaternion 
implementations. What little is here is also incredibly ugly, but implementation 
designs will be explained in the relevant sections.

Generally speaking, this project compares the various uses of SIMD extensions, but 
the implementations make use of GCC's vector extensions rather than the SIMD 
intrinsics provided through `smmintrin.h`. This has a number of implications on the 
requirements to build this project; on X86, SSE and AVX are required, but analogous 
features on other platforms would be required instead.

Implementation correctness is verified by testing against the reference 
implementation using randomly generated sample pairs of dual quaternions.

## Building
The project is set up as a standard out-of-source CMake build, with support for 
`clang` and `gcc` (MSVC not included). 

## Results
The project was configured using the command

`cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native" -B build`

on a Coffee Lake i5-8600k, which supports up to and including AVX2. 

```
$ gcc --version
gcc.exe (Rev3, Built by MSYS2 project) 12.1.0
```

```
$ ./benchmark.exe
sample_count: 10000000
nop_time: 2433µs
reference_time: 77586µs
simd_quaternion_time: 42616µs
matrix_quaternion_time: 36078µs
parallel_dual_quaternion_time: 46296µs
```
(clang produced similar numbers)

`nop_time` provides an estimate on the lower bound for benchmarking overhead, but 
this value is not subtracted off the durations printed. 

Entries which are listed as a non-dual quaternion are dual quaternion 
implementations created by wrapping that quaternion implementation. 

## Implementation Designs

### reference_quaternion
The quaternion product is implemented based off the [Hamilton product](https://en.wikipedia.org/wiki/Quaternion#Hamilton_product).

To my knowledge, glm implements quaternions in this way.

### simd_quaternion
The quaternion product is implemented by viewing quaternions as a sum between 
[scalar and vector parts](https://en.wikipedia.org/wiki/Quaternion#Scalar_and_vector_parts).

For the quaternion product `(s + u) * (t + v)`, for scalars `s, t` and vectors 
`u, v` the produced scalar component of this implementation is 
`(s * t + t * s) - (s * t + <u, v>)` where `<u, v>` is the dot product between `u` 
and `v`, which saves on selecting the vector parts by clearing the scalar component when performing the scalar and dot products.

It is possible that this implementation can be further optimized by changing the 
layout from `(vector, scalar)` to `(scalar, vector)` so that the scalar component 
can be broadcasted using `vbroadcastss` (if available). 

### matrix_quaternion
The quaternion product is implemented by effectively multiplying a matrix by a 
vector. The implementation need not compute the matrix itself, which can save on 
some registers.

### parallel_dual_quaternion
The dual quaternion product is implemented by performing all three quaternion 
products simultaneously. With only AVX support, two 256-bit registers are needed for
accumulating the results. It is doubtful that AVX-512 would significantly improve 
speeds here, because of the hit to core frequency that AVX-512 operations would 
incur.

## Is It Worth Worrying About?
For most applications, the choice of implementation would not significantly impact 
performance, with a few points to consider.
 - Quaternion products via scalar and vector parts is simple to implement.
 - Quaternion products via matrix-vector multiplication involves little wizardry 
   outside of performantly negating only parts of a vector.
 - The Hamilton product might not play well with GPU shaders, although this would 
   come down to how well the compiler can optimize the operation.
 - Building a matrix in a shader to take advantage of specialized matrix-vector 
   multiplication circuitry may negatively impact GPU occupancy. 

Accuracy is also a point to consider. It is consided good practice to re-normalize
your quaternions during long product chains, but only occasionally due to the 
relatively expensive operations involved. A faster implementation might offset the 
cost so that more frequent re-normalization is affordable or make viable strategies
such as an adaptation of Kahann summation to reduce computational error in the first
place, although the degree to which Kahann summation specifically would assist in 
reducing error has yet to be seen.