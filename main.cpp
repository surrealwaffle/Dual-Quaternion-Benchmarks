#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cmath>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <vector>

#include "quaternion.hpp"

// Used for creating test samples
using rng_engine = std::default_random_engine;
using benchmark_duration = std::chrono::microseconds;

struct benchmark_result
{
   benchmark_duration nop_time;
   benchmark_duration reference_time;
   benchmark_duration simd_quaternion_time;
   benchmark_duration matrix_quaternion_time;
   benchmark_duration parallel_dual_quaternion_time;
};

// Performs a benchmark on the implementations.
benchmark_result do_benchmark(int sample_count);

// Tests a dual quaternion Implementation's operator*.
// Returns true if the tests passed, otherwise false.
template<class Implementation>
bool test_implementation_products();

int main(int argc, char* argv[])
{
   int sample_count = 10'000'000;
   if (argc > 1)                           sample_count = std::atoi(argv[1]);
   else if (argc > 2 || sample_count <= 0) return EXIT_FAILURE;
   
   // Pre-test implementations
   if constexpr (test_against_reference)
   {
      std::cout << "Testing implementations products" << std::endl;
      
      if (!test_implementation_products<simd_dual_quaternion>())
         return EXIT_FAILURE;
      if (!test_implementation_products<matrix_dual_quaternion>())
         return EXIT_FAILURE;
      if (!test_implementation_products<parallel_dual_quaternion>())
         return EXIT_FAILURE;
      
      std::cout << "... Done" << std::endl;
   }
   
   benchmark_result result {};
   try
   {
      result = do_benchmark(sample_count);
   } catch (...)
   {
      std::cout << "benchmark failed\n";
   }
   
   std::cout
      << "sample_count: " << sample_count << "\n"
      << "nop_time: " << result.nop_time << "\n"
      << "reference_time: " << result.reference_time << "\n"
      << "simd_quaternion_time: " << result.simd_quaternion_time << "\n"
      << "matrix_quaternion_time: " << result.matrix_quaternion_time << "\n"
      << "parallel_dual_quaternion_time: " << result.parallel_dual_quaternion_time << "\n";
   
   return EXIT_SUCCESS;
}

// Performs a benchmark of a specific implementation.
template<class DualQuaternion>
benchmark_duration bench_implementation(int sample_count, auto sampler);

// Convenient float sample generator.
auto make_sampler()
{
   std::random_device rd;
   std::default_random_engine eng(rd());
   std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
   
   return [eng, dist] () mutable { return dist(eng); };
}

benchmark_result do_benchmark(const int sample_count)
{
   auto sampler = make_sampler();
   
   return {
      .nop_time = bench_implementation<nop_dual_quaternion>(
         sample_count,
         sampler
      ),
      .reference_time = bench_implementation<reference_dual_quaternion>(
         sample_count,
         sampler
      ), 
      .simd_quaternion_time = bench_implementation<simd_dual_quaternion>(
         sample_count,
         sampler
      ),
      .matrix_quaternion_time = bench_implementation<matrix_dual_quaternion>(
         sample_count,
         sampler
      ),
      .parallel_dual_quaternion_time = bench_implementation<parallel_dual_quaternion>(
         sample_count,
         sampler
      ),
   };
}

template<class DualQuaternion>
std::vector<DualQuaternion> generate_samples(int sample_count, auto sampler)
{
   std::vector<DualQuaternion> result;
   result.reserve(sample_count);
   
   std::generate_n(std::back_inserter(result), sample_count, [&sampler] 
   {
      return DualQuaternion::from_gen(std::ref(sampler));
   });
   
   return result;
}

// Checks the product of p * q against product using the reference implementation.
// Returns true if they agree, otherwise false
template<class DualQuaternionInst>
bool check_product_against_reference(
   DualQuaternionInst p, DualQuaternionInst q,
   DualQuaternionInst product,
   float tolerance = 0.1f)
{
   const auto p_ref = convert_to_reference_dq(p);
   const auto q_ref = convert_to_reference_dq(q);
   const auto product_ref = p_ref * q_ref;
   
   const bool nearly_eq = test_dual_quaternions_eq(product, product_ref, tolerance);
   if (!nearly_eq)
   {
      std::cout
         << "For implementation " << DualQuaternionInst::name
         << "\nProduct of: \n\t" << p << "\n\tand\n\t" << q
         << "\n\tGot      " << product
         << "\n\tExpected " << product_ref
         << "\n";
   }
   
   return nearly_eq;
}

template<class DualQuaternion>
benchmark_duration bench_implementation(const int sample_count, auto sampler)
{
   const auto left_samples  = generate_samples<DualQuaternion>(
      sample_count, std::ref(sampler)
   );
   const auto right_samples = generate_samples<DualQuaternion>(
      sample_count, std::ref(sampler)
   );
   
   // This doesn't measure CPU time but there is no clock for that in <chrono>
   using clock = std::chrono::steady_clock;
   
   const auto start = clock::now();
   auto first1 = left_samples.begin();
   auto last1  = left_samples.end();
   auto first2 = right_samples.begin();
   for (; first1 != last1; ++first1, ++first2)
   {
      const DualQuaternion& left = *first1;
      const DualQuaternion& right = *first2;
      
      const auto& product = left * right;
      asm volatile("" : : "rm" (product)); // prevent GCC from optimizing it out
      if constexpr (test_against_reference && is_testable<DualQuaternion>)
      {
         const auto real_product = const_cast<const DualQuaternion&>(product);
         if (!check_product_against_reference(left, right, real_product))
            throw std::runtime_error("Implementation may be faulty");
      }
   }
   const auto end = clock::now();
   
   return std::chrono::duration_cast<benchmark_duration>(end - start);
}

template<class Implementation>
bool test_implementation_products()
{
   if constexpr (is_testable<Implementation>)
   {
      const int runs = 100'000;
      auto sampler = make_sampler();
      
      for (int i = 0; i < runs; ++i)
      {
         const auto left_test = Implementation::from_gen(std::ref(sampler));
         const auto right_test = Implementation::from_gen(std::ref(sampler));
         const auto prod_test = left_test * right_test;
         
         if (!check_product_against_reference(left_test, right_test, prod_test))
         {
            return false;
         }
      }
   }
   
   return true;
}