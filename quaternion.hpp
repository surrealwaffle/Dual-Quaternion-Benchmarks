#pragma once

#include <cstddef>
#include <cstdint>

#include <array>
#include <iterator>
#include <ostream>
#include <functional>
#include <type_traits>
#include <utility>

#include "config.hpp"

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif // __SSE4_1__

// Note: If a quaternion q = w + x * i + y * j + z * k, then an operation that 
// serializes from or to q is done in the order x, y, z, w.

// Support vector types and operations

// notation/order ijkw
typedef float __attribute__((vector_size(4 * sizeof(float)), aligned(4 * sizeof(float)))) v4sf;
typedef int __attribute__((vector_size(4 * sizeof(int)), aligned(4 * sizeof(int)))) v4si;
typedef std::uint32_t __attribute__((vector_size(4 * sizeof(std::uint32_t)), aligned(4 * sizeof(std::uint32_t)))) v4su;

typedef float __attribute__((vector_size(8 * sizeof(float)), aligned(8 * sizeof(float)))) v8sf;
typedef int __attribute__((vector_size(8 * sizeof(int)), aligned(8 * sizeof(int)))) v8si;
typedef std::uint32_t __attribute__((vector_size(8 * sizeof(std::uint32_t)), aligned(8 * sizeof(std::uint32_t)))) v8su;

BENCH_CONSTEXPR float hsum(const v4sf VECTOR_PARAM_REFERENCE u) noexcept
{
   const auto shuf = __builtin_shufflevector(u, u, 1, -1, 3, -1);
   const auto sums = u + shuf;
   return (sums + __builtin_shufflevector(sums, sums, 2, -1, -1, -1))[0];
}

BENCH_CONSTEXPR float dot(
   const v4sf VECTOR_PARAM_REFERENCE u,
   const v4sf VECTOR_PARAM_REFERENCE v) noexcept
{
#ifndef __SSE4_1__
   return hsum(u * v);
#else
   if (std::is_constant_evaluated())
   {
      return hsum(u * v);
   } else 
   {
      // GCC/clang omit the temporaries
      alignas(v4sf) const float uf[] {u[0], u[1], u[2], u[3]};
      alignas(v4sf) const float vf[] {v[0], v[1], v[2], v[3]};
      
      return _mm_cvtss_f32(_mm_dp_ps(
         _mm_load_ps(+uf),
         _mm_load_ps(+vf),
         0xF1 // select all, store into first single
      ));
   }
#endif // __SSE4_1__
}

// Specialized dot product for specific use-case in simd_quaternion
BENCH_CONSTEXPR v4sf simd_quaternion_dot(
   const v4sf VECTOR_PARAM_REFERENCE u,
   const v4sf VECTOR_PARAM_REFERENCE v) noexcept
{
#ifndef __SSE4_1__
   return v4sf{0, 0, 0, hsum(u * v)};
#else
   if (std::is_constant_evaluated())
   {
      return v4sf{0, 0, 0, hsum(u * v)};
   } else 
   {
      // GCC/clang omit the temporaries
      alignas(v4sf) const float uf[] {u[0], u[1], u[2], u[3]};
      alignas(v4sf) const float vf[] {v[0], v[1], v[2], v[3]};
      
      const auto result = _mm_dp_ps(_mm_load_ps(+uf), _mm_load_ps(+vf), 0xF8);
      
      alignas(v4sf) float buf[4];
      _mm_store_ps(+buf, result);
      return v4sf{buf[0], buf[1], buf[2], buf[3]};
   }
#endif // __SSE4_1__
}
   

// result w component is mathematically 0
// as an expression, it is u[3] * v[3] - v[3] * u[3]
BENCH_CONSTEXPR v4sf cross(
   const v4sf VECTOR_PARAM_REFERENCE u, 
   const v4sf VECTOR_PARAM_REFERENCE v) noexcept
{
   const v4sf t0 = __builtin_shufflevector(u, u, 1, 2, 0, 3);
   const v4sf t1 = __builtin_shufflevector(v, v, 1, 2, 0, 3);
   const auto tmp = u * t1 - v * t0;
   return __builtin_shufflevector(tmp, tmp, 1, 2, 0, 3);
}

template<bool... NegateIndicator>
   requires (sizeof...(NegateIndicator) == 4)
BENCH_CONSTEXPR v4sf select_negate(const v4sf VECTOR_PARAM_REFERENCE x) noexcept
{
   return [&x] <int... I> (std::integer_sequence<int, I...>) noexcept
   {
      return __builtin_shufflevector(x, -x, I + 4 * NegateIndicator...);
   }(std::make_integer_sequence<int, 4>{});
}

template<bool... NegateIndicator>
   requires (sizeof...(NegateIndicator) == 8)
BENCH_CONSTEXPR v8sf select_negate(const v8sf VECTOR_PARAM_REFERENCE x) noexcept
{
   return [&x] <int... I> (std::integer_sequence<int, I...>) noexcept
   {
      return __builtin_shufflevector(x, -x, I + 8 * NegateIndicator...);
   }(std::make_integer_sequence<int, 8>{});
}

// Basic quaternion implementation in xyzw order.
// No (explicit) SIMD, but the components are aligned to a 16-byte boundary.
struct reference_quaternion
{
   static constexpr auto name = "reference_quaternion";
   
   alignas(4 * sizeof(float)) float components[4];
   
   template<typename Generator>
   static constexpr reference_quaternion from_gen(Generator gen) noexcept
   {
      return {gen(), gen(), gen(), gen()};
   }
   
   template<class Iterator>
   constexpr Iterator output_to(Iterator it) const noexcept
   {
      *it++ = components[0];
      *it++ = components[1];
      *it++ = components[2];
      *it++ = components[3];
      return it;
   }
};

constexpr reference_quaternion operator+(
   cqarg<reference_quaternion> a,
   cqarg<reference_quaternion> b) noexcept
{
   const auto& [a_x, a_y, a_z, a_w] = a.components;
   const auto& [b_x, b_y, b_z, b_w] = b.components;
   
   return {a_x + b_x, a_y + b_y, a_z + b_z, a_w + b_w};
}

constexpr reference_quaternion operator*(
   cqarg<reference_quaternion> a,
   cqarg<reference_quaternion> b) noexcept
{
   const auto& [a_x, a_y, a_z, a_w] = a.components;
   const auto& [b_x, b_y, b_z, b_w] = b.components;
   
   // in scalar + vector form, a = s + u, b = t + v...
   // a * b = (s * t - <u, v>) + (s * v + t * u + u X v)
   // where <u, v> is the dot product between u and v
   // and u X v is the cross product between u and v
   return {
      a_w * b_x + b_w * a_x + a_y * b_z - a_z * b_y,
      a_w * b_y + b_w * a_y + a_z * b_x - a_x * b_z,
      a_w * b_z + b_w * a_z + a_x * b_y - a_y * b_x,
      a_w * b_w - (a_x * b_x + a_y * b_y + a_z * b_z),
   };
}

// Implements the quaternion product by making use of SIMD extensions
// Strictly speaking, this is the wrong way to use SIMD - operations such as 
// a dot product are parallelized. Throughput should not be much greater than the 
// reference implementation, if at all.
// On the other hand, for a GPU where there is hardware dedicated for swizzling, 
// cross products, and dot products, this implementation or a variant thereof might 
// win out.
struct simd_quaternion
{
   static constexpr auto name = "simd_quaternion";
   
   v4sf components;
   
   friend constexpr simd_quaternion operator+(
      cqarg<simd_quaternion> p, 
      cqarg<simd_quaternion> q) noexcept
   {
      return {p.components + q.components};
   }
   
   friend BENCH_CONSTEXPR simd_quaternion operator*(
      cqarg<simd_quaternion> p, 
      cqarg<simd_quaternion> q) noexcept
   {
      // p := s + u, s in R, u in R^3
      // q := t + v, t in R, v in R^3

      // scalar of cross(p.components, q.components) is 0 for most finite values
      [[maybe_unused]] const auto pcomp = p.components;
      [[maybe_unused]] const auto qcomp = q.components;
      return
      {
         // in {vector, real} format
         cross(pcomp, qcomp)                 // {u X v, 0} for most finite values
         + p.real_part() * qcomp             // + {s * v, s * t}
         + q.real_part() * pcomp             // + {t * u, t * s}
         - simd_quaternion_dot(pcomp, qcomp) // - {0, <u, v> + s * t}
      };
   }
   
   template<typename Generator>
   static constexpr simd_quaternion from_gen(Generator gen)
   {
      return {v4sf{gen(), gen(), gen(), gen()}};
   }
   
   template<class Iterator>
   constexpr Iterator output_to(Iterator it) const noexcept
   {
      *it++ = components[0];
      *it++ = components[1];
      *it++ = components[2];
      *it++ = components[3];
      return it;
   }
   
protected:
   BENCH_CONSTEXPR float real_part() const noexcept { return components[3]; }
   BENCH_CONSTEXPR v4sf vector_part() const noexcept
   {
      constexpr v4sf zero {0, 0, 0, 0};
      return __builtin_shufflevector(components, zero, 0, 1, 2, 4);
   }
}; 

// This implementation views quaternion products as a matrix-vector product.
// In theory, this should yield better throughput than simd_quaternion, but likely 
// not by much.
// A potentially better implementation would represent a quaternion as a matrix,
// but this would come at the cost of a more expensive copy or update operation.
// Considering that quaternions are usually manipulated with value semantics, this 
// might optimize too far into the quaternion product.
struct matrix_quaternion
{
   static constexpr auto name = "matrix_quaternion";
   
   v4sf components;
   
   friend constexpr matrix_quaternion operator+(
      cqarg<matrix_quaternion> p, 
      cqarg<matrix_quaternion> q) noexcept
   {
      return {p.components + q.components};
   }
   
   friend BENCH_CONSTEXPR matrix_quaternion operator*(
      cqarg<matrix_quaternion> p,
      cqarg<matrix_quaternion> q) noexcept
   {
      /*
         With
            p = p_w + p_x * i + p_y * j + p_z * k
            q = q_w + q_x * i + q_y * j + q_z * k
            r = p * q = r_w + r_x * i + r_y * j + r_z * k
         
         [ p_w -p_z  p_y  p_x]   [q_x]   [r_x]
         [ p_z  p_w -p_x  p_y] * [q_y] = [r_y]
         [-p_y  p_x  p_w  p_z]   [q_z]   [r_z]
         [-p_x -p_y -p_z  p_w]   [q_w]   [r_w]
      */
      
      // TODO: implement this better 
      const auto& pv = p.components;
      const auto& qv = q.components;
      const auto [q_x, q_y, q_z, q_w] = qv;
      return 
      {
         // 4+* indicates the negative was selected
         q_x * __builtin_shufflevector(pv, -pv, 3  , 2  , 4+1, 4+0) +
         q_y * __builtin_shufflevector(pv, -pv, 4+2, 3  , 0  , 4+1) +
         q_z * __builtin_shufflevector(pv, -pv, 1  , 4+0, 3  , 4+2) +
         q_w * pv
      };
   }
   
   template<typename Generator>
   static constexpr matrix_quaternion from_gen(Generator gen)
   {
      return {v4sf{gen(), gen(), gen(), gen()}};
   }
   
   template<class Iterator>
   constexpr Iterator output_to(Iterator it) const noexcept
   {
      const auto [w, x, y, z] = components;
      
      *it++ = components[0];
      *it++ = components[1];
      *it++ = components[2];
      *it++ = components[3];
      return it;
   }
};

// Basic template for a dual quaternion built over a Quaternion type
template<class Quaternion>
struct alignas(32) dual_quaternion
{
   static constexpr auto name = Quaternion::name;
   
   Quaternion real;
   Quaternion dual; 
   
   template<typename Generator>
   static constexpr dual_quaternion from_gen(Generator gen) noexcept
   {
      return
      {
         Quaternion::from_gen(std::ref(gen)), 
         Quaternion::from_gen(std::ref(gen))
      };
   }
   
   template<class Iterator>
   constexpr Iterator output_to(Iterator it) const noexcept
   {
      it = real.output_to(it);
      return dual.output_to(it);
   }
   
   template<class Traits>
   friend std::basic_ostream<char, Traits>& operator<<(
      std::basic_ostream<char, Traits>& os,
      const dual_quaternion& dq)
   {
      float buf[8];
      dq.output_to(+buf);
      
      return os
         << "(" << buf[3] << " + [" << buf[0] << "," << buf[1] << "," << buf[2] << "])"
         << " + "
         << "(" << buf[7] << " + [" << buf[4] << "," << buf[5] << "," << buf[6] << "])ε";
   }
};

struct alignas(32) nop_dual_quaternion
{
   static constexpr auto name = "nop_dual_quaternion";
   using do_not_validate = void;
   
   int i;
   
   template<typename Generator>
   static constexpr nop_dual_quaternion from_gen(Generator) noexcept { return {}; }
   
   friend constexpr nop_dual_quaternion operator+(
      cdqarg<nop_dual_quaternion> p,
      cdqarg<nop_dual_quaternion> q) noexcept { return {}; }
   
   friend constexpr nop_dual_quaternion operator*(
      cdqarg<nop_dual_quaternion> p,
      cdqarg<nop_dual_quaternion> q) noexcept { return {}; }
   
};

/*
// STUFF WE ARE NOT TESTING, HERE FOR POSTERITY
template<class Quaternion>
constexpr dual_quaternion<Quaternion> conjugate(
   cqarg<dual_quaternion<Quaternion>> q) noexcept
{
   return {conjugate(q.real), conjugate(q.dual)};
}

template<class Quaternion>
constexpr dual_quaternion<Quaternion> operator+(
   cqarg<dual_quaternion<Quaternion>> p, 
   cqarg<dual_quaternion<Quaternion>> q) noexcept
{
   return {p.real + q.real, p.dual + q.dual};
}

template<class Quaternion>
constexpr dual_quaternion<Quaternion> operator-(
   cqarg<dual_quaternion<Quaternion>> p,
   cqarg<dual_quaternion<Quaternion>> q) noexcept
{
   return {p.real - q.real, p.dual - q.dual};
}
*/

template<class Quaternion>
constexpr dual_quaternion<Quaternion> operator*(
   cdqarg<dual_quaternion<Quaternion>> p,
   cdqarg<dual_quaternion<Quaternion>> q) noexcept
{
   return {p.real * q.real, p.real * q.dual + p.dual * q.real};
}

using reference_dual_quaternion = dual_quaternion<reference_quaternion>;
using simd_dual_quaternion = dual_quaternion<simd_quaternion>;
using matrix_dual_quaternion = dual_quaternion<matrix_quaternion>;

// This type of dual quaternion is intrusive and does not compose quaternions.
// Instead, it views the product of two dual quaternions as the product of three 
// pairs of quaternions, plus some addition.
struct parallel_dual_quaternion
{
   static constexpr auto name = "parallel_dual_quaternion";
   
   v8sf components; // order is ijkw real-dual pairs
   
   template<class Traits>
   friend std::basic_ostream<char, Traits>& operator<<(
      std::basic_ostream<char, Traits>& os,
      const parallel_dual_quaternion& dq)
   {
      float buf[8];
      dq.output_to(+buf);
      
      return os
         << "(" << buf[3] << " + [" << buf[0] << "," << buf[1] << "," << buf[2] << "])"
         << " + "
         << "(" << buf[7] << " + [" << buf[4] << "," << buf[5] << "," << buf[6] << "])ε";
   }
   
   template<typename Generator>
   static constexpr parallel_dual_quaternion from_gen(Generator gen)
   {
      const auto x = gen();
      const auto y = gen();
      const auto z = gen();
      const auto w = gen();
      const auto dx = gen();
      const auto dy = gen();
      const auto dz = gen();
      const auto dw = gen();
      return {v8sf{x, dx, y, dy, z, dz, w, dw}};
   }
   
   template<typename Iterator>
   constexpr Iterator output_to(Iterator it) const 
   {
      const auto [x,dx,y,dy,z,dz,w,dw] = components;
      
      const auto serial_values = {x,y,z,w,dx,dy,dz,dw};
      for (auto v : serial_values)
         *it++ = v;
      return it;
   }
   
   friend constexpr parallel_dual_quaternion operator+(
      cdqarg<parallel_dual_quaternion> p,
      cdqarg<parallel_dual_quaternion> q) noexcept
   {
      return {p.components + q.components};
   }
   
   friend BENCH_CONSTEXPR parallel_dual_quaternion operator*(
      cdqarg<parallel_dual_quaternion> p,
      cdqarg<parallel_dual_quaternion> q) noexcept
   {
      // For quaternions A, B, C, D
      // p := A + Bε, q:= C + Dε
      // p * q = (A * C) + (A * D + B * C)ε
      
      // left:  A, A, B, ??
      // right: C, D, C, ??
      // 0 is the first element since we're shuffling and adding at the end anyway
      
      const auto& pc = p.components;     
      const auto& qc = q.components;

#define ELEMx 0
#define ELEMy 2
#define ELEMz 4
#define ELEMw 6

#define ELEMsink 3

#define SELECT_LEFT(i) __builtin_shufflevector(pc,pc, (i),(i),(i)+1,-1,(i),(i),(i)+1,-1)
#define SELECT_RIGHT(i,j) __builtin_shufflevector(qc,qc, (i),(i)+1,(i),-1,(j),(j)+1,(j),-1)
#define NEGATE_FIRST(x) select_negate<1,1,1,1,0,0,0,0>((x))
#define NEGATE_LAST(x)  select_negate<0,0,0,0,1,1,1,1>((x))

      const v8sf left_x = SELECT_LEFT(ELEMx);
      const v8sf left_y = SELECT_LEFT(ELEMy);
      const v8sf left_z = SELECT_LEFT(ELEMz);
      const v8sf left_w = SELECT_LEFT(ELEMw);
      
      auto r_xy = left_w * SELECT_RIGHT(ELEMx, ELEMy);
      auto r_zw = left_w * SELECT_RIGHT(ELEMz, ELEMw);
      
      r_xy += NEGATE_LAST(left_x * SELECT_RIGHT(ELEMw, ELEMz));
      r_zw += NEGATE_LAST(left_x * SELECT_RIGHT(ELEMy, ELEMx));
      
      r_xy += left_y * SELECT_RIGHT(ELEMz,ELEMw);
      r_zw -= left_y * SELECT_RIGHT(ELEMx,ELEMy);
      
      // r_xy += NEGATE_FIRST(left_z * SELECT_RIGHT(ELEMy,ELEMx));
      r_xy -= NEGATE_LAST(left_z * SELECT_RIGHT(ELEMy,ELEMx));
      r_zw += NEGATE_LAST(left_z * SELECT_RIGHT(ELEMw,ELEMz));
      
      r_xy[ELEMsink] = 0;
      r_zw[ELEMsink] = 0;
      
      r_xy += __builtin_shufflevector(r_xy,r_xy, ELEMsink,2,-1,-1,ELEMsink,4+2,-1,-1);
      r_zw += __builtin_shufflevector(r_zw,r_zw, ELEMsink,2,-1,-1,ELEMsink,4+2,-1,-1);
      
      const auto result = __builtin_shufflevector(r_xy,r_zw, 0,1,4+0,4+1,8+0,8+1,12+0,12+1);
      return {result};

#undef ELEMx
#undef ELEMy
#undef ELEMz
#undef ELEMw
#undef ELEMsink
#undef SELECT_LEFT
#undef SELECT_RIGHT
#undef NEGATE_FIRST
#undef NEGATE_LAST
   }
};

template<class Implementation, class From>
constexpr Implementation convert_dual_quaternion(const From& dq)
{
   float buf[8];
   dq.output_to(buf);
   return Implementation::from_gen(
      [it = +buf] () mutable -> float { return *it++; }
   );
}

template<class From>
constexpr reference_dual_quaternion convert_to_reference_dq(const From& dq)
{
   return convert_dual_quaternion<reference_dual_quaternion>(dq);
}

// Tests if two quaternions are pointwise-equivalent to within an absolute tolerance
// Returns true if they are equivalent as above, otherwise false.
template<class Implementation1, class Implementation2>
constexpr bool test_dual_quaternions_eq(
   const Implementation1& dq1,
   const Implementation2& dq2,
   float tolerance = 0.1f)
{
   float buf1[8];
   float buf2[8];
   
   dq1.output_to(+buf1);
   dq2.output_to(+buf2);
   
   auto first1 = std::begin(buf1);
   auto last1  = std::end(buf1);
   auto first2 = std::begin(buf2);
   
   for (; first1 != last1; ++first1, ++first2)
   {
      const float f1 = *first1;
      const float f2 = *first2;
      
      if (std::abs(f1 - f2) > tolerance)
         return false;
   }
   
   return true;
}

template<typename T>
concept is_testable = !requires {typename T::do_not_validate;};