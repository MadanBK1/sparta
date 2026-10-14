/* Stable free-molecular exposure estimate for a spherical grain.
   P = 1-exp(-lambda), lambda=nu*dt, for Poisson encounters.
   No DSMC particles are modified. The probability describes one or more
   physical impacts on the entire grain, NOT per-DSMC-particle probability.
*/
#ifndef SPARTA_GRAIN_EXPOSURE_REFERENCE_H
#define SPARTA_GRAIN_EXPOSURE_REFERENCE_H
#include "grain_maxwellian_reference.h"
#include "grain_poisson_reference.h"
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS { namespace GrainExposureReference {
struct Exposure {double mean_impacts; double probability_impact;};
inline Exposure evaluate(double n, double radius, double relative_drift,
                        double thermal_sigma, double dt) {
  const double mean=GrainMaxwellianReference::expected_impacts(
       n,radius,relative_drift,thermal_sigma,dt);
  if(!std::isfinite(mean))
    throw std::overflow_error("grain exposure mean overflow");
  return {mean,GrainPoissonReference::probability_at_least_one(mean)};
}
}} // namespaces
#endif
