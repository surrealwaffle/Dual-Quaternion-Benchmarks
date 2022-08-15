#pragma once

// Uncomment to make quaternions pass by reference.
// #define PASS_QUATERNIONS_BY_REFERENCE

// Uncomment to make vectors pass by reference.
// #define PASS_VECTORS_BY_REFERENCE

// Uncomment to make dual quaternions pass by reference.
// #define PASS_DUAL_QUATERNIONS_BY_REFERENCE

#ifdef PASS_QUATERNIONS_BY_REFERENCE
#  define QUATERNION_PARAM_REFERENCE &
#else
#  define QUATERNION_PARAM_REFERENCE 
#endif // PASS_QUATERNIONS_BY_REFERENCE


#ifdef PASS_VECTORS_BY_REFERENCE
#  define VECTOR_PARAM_REFERENCE &
#else
#  define VECTOR_PARAM_REFERENCE 
#endif // PASS_QUATERNIONS_BY_REFERENCE

#ifdef PASS_DUAL_QUATERNIONS_BY_REFERENCE
#  define DUAL_QUATERNION_PARAM_REFERENCE &
#else
#  define DUAL_QUATERNION_PARAM_REFERENCE 
#endif // PASS_QUATERNIONS_BY_REFERENCE

template<typename T> using qarg  = T QUATERNION_PARAM_REFERENCE;
template<typename T> using cqarg = qarg<const T>;

template<typename T> using vecarg  = T VECTOR_PARAM_REFERENCE;
template<typename T> using cvecarg = vecarg<const T>;

template<typename T> using dqarg  = T DUAL_QUATERNION_PARAM_REFERENCE;
template<typename T> using cdqarg = dqarg<const T>;

// Set this to true to test an implementation against the reference implementation.
// constexpr bool test_against_reference = true;
constexpr bool test_against_reference = false;