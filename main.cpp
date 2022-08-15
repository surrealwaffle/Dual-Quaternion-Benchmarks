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

using benchmark_duration = std::chrono::microseconds;

struct benchmark_result
{
   benchmark_duration reference_time;
   benchmark_duration simd_quaternion_time;
   benchmark_duration simd_parallel_time;
};

benchmark_result do_benchmark(int sample_count);

reference_dual_quaternion convert_to_reference_dq(const auto& dq)
{
   float buf[8];
   dq.output_to(+buf);
   dq.output_to(+buf + 4);
   auto read = [it = +buf] () mutable -> float { return *it++; };
   return reference_dual_quaternion::from_gen(read);
}

template<class Implementation>
bool test_dq_implementation();

int main(int argc, char* argv[])
{
   int sample_count = 10'000'000;
   if (argc > 1)                           sample_count = std::atoi(argv[1]);
   else if (argc > 2 || sample_count <= 0) return EXIT_FAILURE;
   
   // Pre-test implementations
   if constexpr (test_against_reference)
   {
      std::cout << "Testing various implementations" << std::endl;
      
      if (!test_dq_implementation<simd_dual_quaternion>())
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
      << "reference_time: " << result.reference_time << "\n"
      << "simd_quaternion_time: " << result.simd_quaternion_time << "\n"
      << "simd_parallel_time: " << result.simd_parallel_time << "\n";
   
   return EXIT_SUCCESS;
}

template<class DualQuaternion>
benchmark_duration bench_implementation(int sample_count, auto sampler);

auto make_sampler()
{
   std::random_device rd;
   std::default_random_engine eng(rd());
   std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
   
   return [eng, dist] () mutable { return dist(eng); };
}

bool test_dqs(auto dq1, auto dq2, float tolerance = 0.1f)
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
      const float s = *first1;
      const float t = *first2;
      if (std::abs(s - t) > tolerance)
         return false;
   }
   
   return true;
}

benchmark_result do_benchmark(const int sample_count)
{
   auto sampler = make_sampler();
   
   return {
      .reference_time = bench_implementation<reference_dual_quaternion>(
         sample_count,
         sampler
      ), 
      .simd_quaternion_time = bench_implementation<simd_dual_quaternion>(
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
bool check_against_reference(
   DualQuaternionInst p, DualQuaternionInst q,
   DualQuaternionInst product,
   float tolerance = 0.1f)
{
   const auto p_ref = convert_to_reference_dq(p);
   const auto q_ref = convert_to_reference_dq(q);
   const auto product_ref = p_ref * q_ref;
   
   const bool nearly_eq = test_dqs(product, product_ref, tolerance);
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
   
   std::vector<DualQuaternion> products;
   products.resize(sample_count); 
   
   // This doesn't measure CPU time but there is no clock for that in <chrono>
   using clock = std::chrono::steady_clock;
   
   const auto start = clock::now();
   std::transform(
      left_samples.begin(), left_samples.end(),
      right_samples.begin(),
      products.begin(),
      [] (cqarg<DualQuaternion> left, cqarg<DualQuaternion> right)
         noexcept(!test_against_reference)
      {
         auto product = left * right;
         
         if constexpr (test_against_reference)
         {
            if (!check_against_reference(left, right, product))
               throw std::runtime_error("Implementation may be faulty");
         }
         
         return product;
      }
   );
   const auto end = clock::now();
   const volatile auto foo = products;
   
   return std::chrono::duration_cast<benchmark_duration>(end - start);
}

template<class Implementation>
bool test_dq_implementation()
{
   const int runs = 100'000;
   auto sampler = make_sampler();
   
   for (int i = 0; i < runs; ++i)
   {
      const auto left_ref  = reference_dual_quaternion::from_gen(sampler);
      const auto left_test = Implementation::from_gen(std::ref(sampler));
      
      const auto right_ref  = reference_dual_quaternion::from_gen(sampler);
      const auto right_test = Implementation::from_gen(std::ref(sampler));
      
      const auto prod_ref  = left_ref * right_ref;
      const auto prod_test = left_test * right_test;
      
      if (!test_dqs(prod_ref, prod_test))
      {
         std::cout
            << "For implementation " << Implementation::name
            << "\nProduct of: \n\t" << left_ref << "\n\tand\n\t" << right_ref
            << "\n\tGot      " << prod_test
            << "\n\tExpected " << prod_ref
            << "\n";
         return false;
      }
   }
   
   return true;
}