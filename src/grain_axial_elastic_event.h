/* Conservative axial impulse applied to a gas particle and one spherical
   grain in an isolated event. This is a physics verification primitive,
   NOT a collision-selection algorithm for axisymmetric DSMC.
   For a DSMC superparticle representing W molecules, the effective
   interacting mass is W*m only if ALL represented molecules take part.
   Partial collisions require particle splitting or a separate statistical
   approximation. No radial component is treated as globally conserved in
   this axisymmetric population model. */
#ifndef SPARTA_GRAIN_AXIAL_ELASTIC_EVENT_H
#define SPARTA_GRAIN_AXIAL_ELASTIC_EVENT_H
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS { namespace GrainAxialElasticEvent {
struct Result { double gas_velocity, grain_velocity, impulse_to_grain; };
inline Result collide(double gas_velocity, double grain_velocity,
                      double molecule_mass, double molecules_per_particle,
                      double grain_mass) {
  if (!std::isfinite(gas_velocity) || !std::isfinite(grain_velocity) ||
      !std::isfinite(molecule_mass) || molecule_mass<=0. ||
      !std::isfinite(molecules_per_particle) || molecules_per_particle<=0. ||
      !std::isfinite(grain_mass) || grain_mass<=0.)
    throw std::invalid_argument("invalid axial elastic collision state");
  const double effective_gas_mass=molecule_mass*molecules_per_particle;
  if (!std::isfinite(effective_gas_mass) || effective_gas_mass<=0.)
    throw std::invalid_argument("invalid effective gas mass");
  const double relative=gas_velocity-grain_velocity;
  // Reduced mass gives robust equal and opposite impulses.
  const double mu=effective_gas_mass*grain_mass/
                   (effective_gas_mass+grain_mass);
  const double impulse=2.*mu*relative;
  return {gas_velocity-impulse/effective_gas_mass,
          grain_velocity+impulse/grain_mass,impulse};
}
}} // namespaces
#endif
