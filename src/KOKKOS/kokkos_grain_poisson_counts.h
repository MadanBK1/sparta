/* Diagnostic-only stochastic physical encounter counts in annular cells.
   Do NOT interpret sampled physical impacts as whole weighted DSMC
   representative collisions or modify the gas/grain velocities.
   Valid for nonnegative finite mean <= 16; bounded event loop returns
   -1 on exhaustion (caller must reject). Counter RNG is deterministic
   per (seed, timestep, representative index) independent of scheduling.
   Hash RNG is for regression/prototyping, NOT validated production DSMC RNG.
*/
#ifndef SPARTA_KOKKOS_GRAIN_POISSON_COUNTS_H
#define SPARTA_KOKKOS_GRAIN_POISSON_COUNTS_H
#include <Kokkos_Core.hpp>
#include <cstdint>
namespace SPARTA_NS { namespace GrainPoissonKokkos {
KOKKOS_INLINE_FUNCTION
uint64_t mix(uint64_t x) {
  x+=UINT64_C(0x9e3779b97f4a7c15);
  x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
  return x^(x>>31);
}
KOKKOS_INLINE_FUNCTION
double uniform_open(uint64_t key,uint64_t counter) {
  const uint64_t bits=mix(key+counter*UINT64_C(0x9e3779b97f4a7c15))>>11;
  return (static_cast<double>(bits)+0.5)*0x1.0p-53;
}
template<class ExecSpace,class MeanView,class CountView>
void sample_counts(const MeanView& means,const CountView& counts,
                   uint64_t seed,uint64_t timestep) {
  const int n=static_cast<int>(means.extent(0));
  Kokkos::parallel_for("grain_poisson_physical_event_counts",
    Kokkos::RangePolicy<ExecSpace>(0,n),KOKKOS_LAMBDA(const int i) {
      const double mean=means(i);
      if (!(mean>=0.0 && mean<=16.0)) {counts(i)=-1;return;}
      const uint64_t key=mix(seed^mix(timestep)^mix(static_cast<uint64_t>(i)));
      double elapsed=0.0;
      // Exponential waiting times; cap prevents unbounded device loops.
      for(int event=0;event<256;++event) {
        elapsed+=-Kokkos::log(uniform_open(key,static_cast<uint64_t>(event)));
        if(elapsed>=mean) {counts(i)=event;return;}
      }
      counts(i)=-1; // extremely unlikely for supported means
    });
}
}} // namespaces
#endif
