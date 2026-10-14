/* Kokkos adapter operating directly on SPARTA's Particle::OnePart view.
   This is a diagnostic axial momentum ledger, NOT a collision operator.
   It reads species mass, physical weighting and particle velocities on
   device without copying particle arrays to host. This demonstrates the
   storage interface required by an eventual validated grain operator.
   Caller must pass device-accessible ParticleKokkos::k_particles.view_device(),
   species masses, grid cell weights already synchronized on device.
   Only axial momentum is summed (axisymmetric radial vectors cancel in
   the aggregate). No grain surface collision selection is attempted.
*/
#ifndef SPARTA_KOKKOS_GRAIN_PARTICLE_MOMENTUM_AUDIT_H
#define SPARTA_KOKKOS_GRAIN_PARTICLE_MOMENTUM_AUDIT_H
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <cmath>
namespace SPARTA_NS { namespace AxisymmetricGrains {
template<class ExecSpace,class ParticleView,class SpeciesMassView>
double axial_gas_momentum(const ParticleView& particles,
                          const SpeciesMassView& species_mass,
                          int nlocal,double fnum,bool cellweightflag) {
  if(nlocal<0 || static_cast<size_t>(nlocal)>particles.extent(0) ||
     !std::isfinite(fnum) || !(fnum>0.))
    throw std::invalid_argument("invalid particle range or fnum");
  double result=0.;
  Kokkos::parallel_reduce("grain_audit_axial_momentum",
    Kokkos::RangePolicy<ExecSpace>(0,nlocal),
    KOKKOS_LAMBDA(const int i,double& sum) {
      const auto p=particles(i);
      const double w=cellweightflag?p.weight:1.0;
      sum += fnum * species_mass(p.ispecies) * w * p.v[0];
    },result);
  return result;
}
}} // namespaces
#endif
