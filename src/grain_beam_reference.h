/* Free-molecular, uniform monoenergetic-beam reference for a sphere.
   Not a SPARTA collision sampler: valid only for a dilute collisionless
   beam and an isolated, fully exposed sphere with relative beam speed.
   The number is an expectation; gas thermal spread, shadowing, local
   gradients, and axisymmetric sampling are NOT included. */
#ifndef SPARTA_GRAIN_BEAM_REFERENCE_H
#define SPARTA_GRAIN_BEAM_REFERENCE_H
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS { namespace GrainBeamReference {
inline double expected_impacts(double number_density, double radius,
                               double relative_speed, double dt) {
  if (!std::isfinite(number_density) || number_density<0. ||
      !std::isfinite(radius) || !(radius>0.) ||
      !std::isfinite(relative_speed) || relative_speed<0. ||
      !std::isfinite(dt) || !(dt>0.))
    throw std::invalid_argument("invalid beam reference parameters");
  const double pi=std::acos(-1.);
  return number_density * pi * radius*radius * relative_speed * dt;
}
inline double expected_simulator_events(double number_density, double radius,
                               double relative_speed, double dt,
                               double physical_molecules_per_event) {
  if (!std::isfinite(physical_molecules_per_event) ||
      !(physical_molecules_per_event>0.))
    throw std::invalid_argument("invalid simulation event weight");
  return expected_impacts(number_density,radius,relative_speed,dt) /
         physical_molecules_per_event;
}
}} // namespaces
#endif
