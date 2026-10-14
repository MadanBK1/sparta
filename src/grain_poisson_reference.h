/* Physical Poisson encounter-count sampler for isolated spherical grains.
   Reference only: expected count must be computed from a suitable local
   relative-speed distribution. No association with a particular SPARTA
   DSMC simulation particle is made here. Never treat Poisson events as
   automatically equivalent to DSMC particle-surface collisions.
*/
#ifndef SPARTA_GRAIN_POISSON_REFERENCE_H
#define SPARTA_GRAIN_POISSON_REFERENCE_H
#include <cmath>
#include <stdexcept>
#include <limits>
namespace SPARTA_NS { namespace GrainPoissonReference {
inline double probability_at_least_one(double expected_events) {
  if (!std::isfinite(expected_events) || expected_events<0.)
    throw std::invalid_argument("invalid Poisson mean");
  return -std::expm1(-expected_events);
}
// Waiting-time method. Exponential interarrival waiting times.
// Uniform generator must return strictly 0<u<1. Work scales as event count;
// useful only for moderate means and for validation, not large GPU rates.
template<class Uniform>
int sample_waiting_time(double mean, Uniform uniform) {
  if (!std::isfinite(mean) || mean<0.)
    throw std::invalid_argument("invalid Poisson mean");
  double elapsed=0.;
  int count=0;
  while (true) {
    double u=uniform();
    if (!(u>0. && u<1.) || !std::isfinite(u))
      throw std::invalid_argument("uniform RNG must be in (0,1)");
    elapsed += -std::log(u);
    if(elapsed>=mean) return count;
    if(count==std::numeric_limits<int>::max())
      throw std::overflow_error("Poisson count overflow");
    ++count;
  }
}
}} // namespaces
#endif
