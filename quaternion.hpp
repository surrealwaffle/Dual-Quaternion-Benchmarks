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

// Note: If a quaternion q = w + x * i + y * j + z * k, then an operation that 
// serializes from or to q is done in the order x, y, z, w.

// Support vector types and operations

// notation/order ijkw
typedef float __attribute__((vector_size(4 * sizeof(float)), aligned(4 * sizeof(float)))) v4sf;
typedef int __attribute__((vector_size(4 * sizeof(int)), aligned(4 * sizeof(int))))   v4si;
typedef std::uint32_t __attribute__((vector_size(4 * sizeof(uint32_t)), aligned(4 * sizeof(uint32_t)))) v4su;

typedef float __attribute__((vector_size(8 * sizeof(float)), aligned(8 * sizeof(float)))) v8sf;
typedef float __attribute__((vector_size(16 * sizeof(float)), aligned(16 * sizeof(float)))) v16sf;
typedef int __attribute__((vector_size(8 * sizeof(int)), aligned(8 * sizeof(int))))   v8si;

constexpr float hsum(const v4sf VECTOR_PARAM_REFERENCE u) noexcept
{
   // There are a few problems using the intrinsics for this operation:
   //  1. not portable to different architectures;
   //  2. not constexpr friendly (not a big issue);
   //  3. seemingly becomes an optimization barrier for GCC
   
#  ifdef __SSE3__
      constexpr bool supports_sse3 = true;
#  else
      constexpr bool supports_sse3 = false;
#  endif // __SSE3__  
   
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

constexpr float dot(
   const v4sf VECTOR_PARAM_REFERENCE u,
   const v4sf VECTOR_PARAM_REFERENCE v) noexcept
{
   return hsum(u * v);
}

// result w component is mathematically 0
// as an expression, it is u[3] * v[3] - v[3] * u[3]
constexpr v4sf cross(
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

consteval v4su make_negate_mask(std::array<bool, 4> ind) noexcept
{
   return [&ind] <std::size_t... I> (std::index_sequence<I...>) noexcept
   {
      return v4su{ind[I]...} << 31;
   }(std::make_index_sequence<ind.size()>{});
}

constexpr v4sf select_negate(v4sf x, const v4su mask)
{
   if (std::is_constant_evaluated())
   {
      for (int i = 0; i < 4; ++i)
      {
         if (mask[i]) x[i] = -x[i];
      }
      return x;
   } else 
   {
      // looks like really bad abuse of reinterpret_cast but its concise
      return reinterpret_cast<v4sf>(reinterpret_cast<v4su&>(x) ^ mask);
   }
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
   
   friend constexpr simd_quaternion operator*(
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
         - v4sf{0, 0, 0, dot(pcomp, qcomp)}  // - {0, <u, v> + s * t}
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
   
   friend constexpr matrix_quaternion operator*(
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
      const auto pv = p.components;
      const auto qv = q.components;
      const auto [q_x, q_y, q_z, q_w] = qv;
      return 
      {
         (
            q_x * select_negate(__builtin_shuffle(pv, v4si{3,2,1,0}), make_negate_mask({0,0,1,1}))
            +
            q_y * select_negate(__builtin_shuffle(pv, v4si{2,3,0,1}), make_negate_mask({1,0,0,1}))
         )
         +
         (
            q_z * select_negate(__builtin_shuffle(pv, v4si{1,0,3,2}), make_negate_mask({0,1,0,1}))
            +
            q_w * pv
         )
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
   
   friend constexpr parallel_dual_quaternion operator*(
      cdqarg<parallel_dual_quaternion> p,
      cdqarg<parallel_dual_quaternion> q) noexcept
   {
      // For quaternions A, B, C, D
      // p := A + Bε, q:= C + Dε
      // p * q = (A * C) + (A * D + B * C)ε
      
      // left:  0, A, B, A
      // right: 0, D, C, C
      // 0 is the first element since we're shuffling and adding at the end anyway
      
      const auto& pcomp = p.components;     
      const auto& qcomp = q.components;

      v4sf left_x = __builtin_shufflevector(pcomp, pcomp, -1, 0, 1, 0);
      left_x[0] = 0;
      
      v4sf left_y = __builtin_shufflevector(pcomp, pcomp, -1, 2, 3, 2);
      left_y[0] = 0;
      
      v4sf left_z = __builtin_shufflevector(pcomp, pcomp, -1, 4, 5, 4);
      left_z[0] = 0;
      
      v4sf left_w = __builtin_shufflevector(pcomp, pcomp, -1, 6, 7, 6);
      left_w[0] = 0;
      
      v4sf right_x = __builtin_shufflevector(qcomp, qcomp, -1, 1, 0, 0);
      right_x[0] = 0;
      
      v4sf right_y = __builtin_shufflevector(qcomp, qcomp, -1, 3, 2, 2);
      right_y[0] = 0;
      
      v4sf right_z = __builtin_shufflevector(qcomp, qcomp, -1, 5, 4, 4);
      right_z[0] = 0;
      
      v4sf right_w = __builtin_shufflevector(qcomp, qcomp, -1, 7, 6, 6);
      right_w[0] = 0;
      
      // Now we follow a quaternion product, building up by component
      // Note: This is still a bad way of doing this.
      v4sf r_x
         = left_w * right_x 
         + left_x * right_w
         + left_y * right_z
         - left_z * right_y;
      
      v4sf r_y
         = left_w * right_y
         - left_x * right_z
         + left_y * right_w
         + left_z * right_x;
      
      v4sf r_z
         = left_w * right_z
         + left_x * right_y
         - left_y * right_x
         + left_z * right_w;
      
      v4sf r_w
         = left_w * right_w
         - left_x * right_x
         - left_y * right_y
         - left_z * right_z;
      
      // Then shuffle and sum
      r_x += __builtin_shufflevector(r_x, r_x, 3, 2, -1, -1);
      r_y += __builtin_shufflevector(r_y, r_y, 3, 2, -1, -1);
      r_z += __builtin_shufflevector(r_z, r_z, 3, 2, -1, -1);
      r_w += __builtin_shufflevector(r_w, r_w, 3, 2, -1, -1);
      
      // Build back up to v8sf
      const v4sf first_part  = __builtin_shufflevector(r_x, r_y, 0, 1, 4, 5);
      const v4sf second_part = __builtin_shufflevector(r_z, r_w, 0, 1, 4, 5);
      const v8sf result = __builtin_shufflevector(first_part, second_part, 0, 1, 2, 3, 4, 5, 6, 7);
      
      return {result};
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