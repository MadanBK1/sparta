/* SPARTA-consistent DSMC representative mass reference.
   This matches the existing rigid recoil expression in
   src/KOKKOS/update_kokkos.cpp:
      msuper=fnum*species_mass;
      if(cellweightflag) msuper*=particle.weight;
   It is an audit/reference helper, not a new weighting convention.
   NOTE: SPARTA's axisymmetric radial resampling and distribution remain
   separate, and this does not validate off-axis 3D grain collisions.
*/
#ifndef SPARTA_GRAIN_DSMC_MASS_REFERENCE_H
#define SPARTA_GRAIN_DSMC_MASS_REFERENCE_H
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS { namespace GrainDSMCMassReference {
inline double represented_mass(double fnum,double species_mass,
                               bool cellweightflag,double particle_weight) {
 if (!std::isfinite(fnum)||!(fnum>0.)||
     !std::isfinite(species_mass)||!(species_mass>0.)||
     (cellweightflag && (!std::isfinite(particle_weight)||
                         !(particle_weight>0.))))
   throw std::invalid_argument("invalid SPARTA representative mass parameters");
 const double mass=fnum*species_mass*(cellweightflag ? particle_weight : 1.0);
 if(!std::isfinite(mass)||!(mass>0.))
   throw std::overflow_error("SPARTA representative mass overflow");
 return mass;
}
inline double axial_momentum_change(double fnum,double species_mass,
                   bool cellweightflag,double particle_weight,
                   double vx_before,double vx_after) {
 if(!std::isfinite(vx_before)||!std::isfinite(vx_after))
   throw std::invalid_argument("invalid axial velocities");
 return represented_mass(fnum,species_mass,cellweightflag,particle_weight)*
        (vx_after-vx_before);
}
}} // namespaces
#endif
