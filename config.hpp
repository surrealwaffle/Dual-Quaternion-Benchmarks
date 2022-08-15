#pragma once

// Define PASS_QUATERNIONS_BY_REFERENCE to make quaternions passed by reference 
// (defaults to pass-by-value).
#ifdef PASS_QUATERNIONS_BY_REFERENCE
template<typename T> using qarg = T&;
#else
template<typename T> using qarg = T;
#endif // PASS_QUATERNIONS_BY_REFERENCE

// Define PASS_VECTORS_BY_REFERENCE to make vectors passed by reference 
// (defaults to pass-by-value).
#ifdef PASS_VECTORS_BY_REFERENCE
template<typename T> using vecarg = T&;
#else
template<typename T> using vecarg = T;
#endif // PASS_QUATERNIONS_BY_REFERENCE

// Define PASS_DUAL_QUATERNIONS_BY_REFERENCE to make dual quaternions passed by 
// reference (defaults to pass-by-value).
#ifdef PASS_DUAL_QUATERNIONS_BY_REFERENCE
template<typename T> using dqarg = T&;
#else
template<typename T> using dqarg = T;
#endif // PASS_DUAL_QUATERNIONS_BY_REFERENCE

template<typename T> using cqarg   = qarg<const T>;
template<typename T> using cvecarg = vecarg<const T>;
template<typename T> using cdqarg  = dqarg<const T>;