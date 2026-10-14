/* Device-side axial-only collision bookkeeping on SPARTA::OnePart views.
   This is an adapter prototype, not a gas-grain collision selection law.
   Each i owns particles(i), so no concurrent particle writes. The event
   probability is a physical-molecule participation fraction, NOT a
   superparticle Bernoulli trial. The averaged velocity destroys velocity
   variance; production DSMC needs a validated splitting/stochastic closure.
   Do not apply to the off-axis radial momentum balance.
*/
#ifndef SPARTA_KOKKOS_GRAIN_ONEPART_AXIAL_EXCHANGE_H
#define SPARTA_KOKKOS_GRAIN_ONEPART_AXIAL_EXCHANGE_H
#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS { namespace AxisymmetricGrains {
template<class ExecSpace,class ParticleView,class SpeciesMassView,
         class ProbabilityView,class TargetView,class AfterVelocityView,
         class ImpulseView>
void onepart_axial_partial_exchange(
    const ParticleView& particles,const SpeciesMassView& species_mass,
    const ProbabilityView& probability,const TargetView& target,
    const AfterVelocityView& conditional_vx,const ImpulseView& impulses,
    int nlocal,double fnum,bool cellweightflag) {
  if(nlocal<0 || static_cast<size_t>(nlocal)>particles.extent(0) ||
     static_cast<size_t>(nlocal)>probability.extent(0) ||
     static_cast<size_t>(nlocal)>target.extent(0) ||
     static_cast<size_t>(nlocal)>conditional_vx.extent(0) ||
     impulses.extent(1)<2 || !std::isfinite(fnum) || !(fnum>0.))
    throw std::invalid_argument("invalid onepart axial exchange arguments");
  // Caller validates per-particle species, grain, probability and velocity.
  // No host-side verification on the hot path.
  Kokkos::parallel_for("grain_onepart_axial_partial_exchange",
      Kokkos::RangePolicy<ExecSpace>(0,nlocal),
      KOKKOS_LAMBDA(const int i) {
        const double p=probability(i);
        if(p==0.) return;
        const int species=particles(i).ispecies;
        const double w=cellweightflag ? particles(i).weight : 1.0;
        const double mass=fnum*species_mass(species)*w;
        const double oldv=particles(i).v[0];
        const double dv=conditional_vx(i)-oldv;
        particles(i).v[0]=oldv+p*dv;
        Kokkos::atomic_add(&impulses(target(i),0),-mass*p*dv);
      });
}
}} // namespaces
#endif
