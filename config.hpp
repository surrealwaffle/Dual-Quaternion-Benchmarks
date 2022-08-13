#pragma once

// The macro PASS_QUATERNIONS_BY_REFERENCE causes quaternions and dual quaternions 
// to be passed by (const) reference rather than (const) value to functions.
#ifdef PASS_QUATERNIONS_BY_REFERENCE
template<typename T> using qarg = T&;
#else
template<typename T> using qarg = T;
#endif // PASS_QUATERNIONS_BY_REFERENCE

template<typename T> using cqarg = qarg<const T>;