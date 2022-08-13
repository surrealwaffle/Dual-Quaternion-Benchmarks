#pragma once

#include <functional>

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
   cqarg<dual_quaternion<Quaternion>> p,
   cqarg<dual_quaternion<Quaternion>> q) noexcept
{
   return {p.real * q.real, p.real * q.dual + p.dual * q.real};
}

using reference_dual_quaternion = dual_quaternion<reference_quaternion>;