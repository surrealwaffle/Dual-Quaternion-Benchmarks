#include <cassert>
#include <cstddef>
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
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

int main(int argc, char* argv[])
{
   int sample_count = 10'000'000;
   if (argc > 1)                           sample_count = std::atoi(argv[1]);
   else if (argc > 2 || sample_count <= 0) return EXIT_FAILURE;
   
   const auto result = do_benchmark(sample_count);
   
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
      [] (cqarg<DualQuaternion> left, cqarg<DualQuaternion> right) noexcept
      {
         return left * right;
      }
   );
   const auto end = clock::now();
   const volatile auto foo = products;
   
   return std::chrono::duration_cast<benchmark_duration>(end - start);
}