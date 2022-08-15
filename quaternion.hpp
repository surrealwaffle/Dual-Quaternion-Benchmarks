#pragma once

#include <functional>
#include <type_traits>

#include "config.hpp"

// Basic quaternion implementation in xyzw order.
// No SIMD, but the components are aligned to a 16-byte boundary.
struct reference_quaternion
{
   alignas(16) float components[4];
   
   static constexpr reference_quaternion zero() noexcept
   {
      return {0, 0, 0, 0};
   }
   
   static constexpr reference_quaternion identity() noexcept
   {
      return {0, 0, 0, 1};
   }
   
   template<typename Generator>
   static constexpr reference_quaternion from_gen(Generator gen) noexcept
   {
      return {gen(), gen(), gen(), gen()};
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
// Strictly speaking, this is the wrong way of using SIMD.
// Whether it's worth it in terms of throughput remains to be seen.
// GCC/clang are required here, but clang is unsupported because it does not support
// GCC's vector builtins. Arguably we could write replacements here, as I expect 
// clang would be able to better optimize the required functions.
struct simd_quaternion
{
   // clang/gcc only, vector of float/int
   
   // notation ijkw
   typedef float __attribute__((vector_size(4 * sizeof(float)), aligned(4 * sizeof(float)))) v4sf;
   typedef int __attribute__((vector_size(4 * sizeof(int)), aligned(4 * sizeof(int))))   v4si;
   
   v4sf components;
   
   friend constexpr simd_quaternion operator+(
      cqarg<simd_quaternion> p, 
      cqarg<simd_quaternion> q) noexcept
   {
      return {p.components + q.components};
   }
   
   friend constexpr simd_quaternion operator*(
      cqarg<simd_quaternion> p, 
      cqarg<simd_quaternion> q) noexcept
   {
      // p := s + u, s in R, u in R^3
      // q := t + v, t in R, v in R^3
      
      const auto pcomp = p.components;
      const auto qcomp = q.components;
      // w component of cross(pcomp, qcomp) is 0 for most finite values
      return
      {
         cross(pcomp, qcomp) // {u X v, 0} for most finite values
         + p.real_part() * qcomp // {s * v, s * t}
         + q.real_part() * pcomp // {t * u, t * s}
         - v4sf{0, 0, 0, dot(pcomp, qcomp)} // -<u, v> - s * t
      };
   }
   
   static constexpr simd_quaternion zero() noexcept
   {
      return {v4sf{0, 0, 0, 0}};
   }
   
   static constexpr simd_quaternion identity() noexcept
   {
      return {v4sf{0, 0, 0, 1}};
   }
   
   template<typename Generator>
   static constexpr simd_quaternion from_gen(Generator gen)
   {
      return {v4sf{gen(), gen(), gen(), gen()}};
   }
   
protected:
   constexpr float real_part() const noexcept { return components[3]; }
   constexpr v4sf vector_part() const noexcept
   {
      auto x = components;
      x[3] = 0;
      return x;
   }

   static constexpr v4sf vector_part(cqarg<simd_quaternion> p) noexcept
   {
      auto x = p.components;
      x[3] = 0;
      return x;
   }

   // Implements a horizontal sum.
   static constexpr float hsum(
      const v4sf VECTOR_PARAM_REFERENCE u) noexcept
   {
      // There are a few problems using the intrinsics for this operation:
      //  1. not portable to different architectures;
      //  2. not constexpr friendly (not a big issue);
      //  3. seemingly becomes an optimization barrier for GCC
      
#     ifdef __SSE3__
         constexpr bool supports_sse3 = true;
#     else
         constexpr bool supports_sse3 = false;
#     endif // __SSE3__  
      
      if (std::is_constant_evaluated() || !supports_sse3)
      {
         const auto shuf = __builtin_shuffle(u, v4si{1, 1, 3, 3});
         const auto sums = u + shuf;
         return (sums + __builtin_shuffle(sums, v4si{2, 3, 2, 3}))[0];
      } else 
      {
         // GCC omits the temporary, but GCC emits very close code to this when 
         // AVX is enabled.
         auto x = u;
         v4sf temp;
         __asm ( // x -> {a, b, c, d}
            "movshdup %[x], %[temp] \n\t" // temp -> {b, b, d, d}
            "addps    %[temp], %[x] \n\t" // x    -> {a+b, b+b, c+d, d+d}
            "movhlps  %[x], %[temp] \n\t" // temp -> {c+d, d+d, d, d}
            "addss    %[temp], %[x] \n\t" // x    -> {(a+b)+(c+d), ...}
            : [x] "+x" (x), [temp] "=x" (temp)
            : // no input-only operands, a is read-and-write
            : // no explicit clobbers needed
         );
         return x[0];
      }
   }
   
   // result w component is mathematically 0
   // as an expression, it is u[3] * v[3] - v[3] * u[3]
   static constexpr v4sf cross(
      const v4sf VECTOR_PARAM_REFERENCE u, 
      const v4sf VECTOR_PARAM_REFERENCE v) noexcept
   {
      const v4sf t0 = __builtin_shuffle(u, v4si{1, 2, 0, 3});
      const v4sf t1 = __builtin_shuffle(v, v4si{1, 2, 0, 3});
      return __builtin_shuffle(
        u * t1 - v * t0,
        v4si{1, 2, 0, 3}
      );
   }
   
   static constexpr float dot(
      const v4sf VECTOR_PARAM_REFERENCE u, 
      const v4sf VECTOR_PARAM_REFERENCE v) noexcept
   {
      return hsum(u * v);
   }
};

// Basic template for a dual quaternion built over a Quaternion type
template<class Quaternion>
struct dual_quaternion
{
   Quaternion real;
   Quaternion dual; 
   
   static constexpr dual_quaternion zero() noexcept
   {
      return {Quaternion::zero(), Quaternion::zero()};
   }
   
   static constexpr dual_quaternion identity() noexcept
   {
      return {Quaternion::identity(), Quaternion::zero()};
   }
   
   template<typename Generator>
   static constexpr dual_quaternion from_gen(Generator gen) noexcept
   {
      return
      {
         Quaternion::from_gen(std::ref(gen)), 
         Quaternion::from_gen(std::ref(gen))
      };
   }
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