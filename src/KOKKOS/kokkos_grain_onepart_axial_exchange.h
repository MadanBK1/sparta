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
// Device-side preflight, with only a scalar error count copied to host.
// This is a debugging/verification path; callers can prevalidate at setup
// and use the unchecked kernel when performance is critical.
template<class ExecSpace,class ParticleView,class SpeciesMassView,
         class ProbabilityView,class TargetView,class AfterVelocityView,
         class ImpulseView>
int count_invalid_onepart_events(
    const ParticleView& particles,const SpeciesMassView& species_mass,
    const ProbabilityView& probability,const TargetView& target,
    const AfterVelocityView& conditional_vx,const ImpulseView& impulses,
    int nlocal,bool cellweightflag) {
  if(nlocal<0 || static_cast<size_t>(nlocal)>particles.extent(0) ||
     static_cast<size_t>(nlocal)>probability.extent(0) ||
     static_cast<size_t>(nlocal)>target.extent(0) ||
     static_cast<size_t>(nlocal)>conditional_vx.extent(0) ||
     impulses.extent(1)<2)
    throw std::invalid_argument("invalid onepart validation dimensions");
  const int nspecies=static_cast<int>(species_mass.extent(0));
  const int ngrains=static_cast<int>(impulses.extent(0));
  int bad=0;
  Kokkos::parallel_reduce("grain_onepart_event_preflight",
    Kokkos::RangePolicy<ExecSpace>(0,nlocal),
    KOKKOS_LAMBDA(const int i,int& local) {
      const double p=probability(i);
      if(!Kokkos::isfinite(p) || p<0. || p>1.) {
        local++;return;
      }
      if(p==0.) return;
      const int s=particles(i).ispecies, g=target(i);
      if(s<0 || s>=nspecies || g<0 || g>=ngrains) {
        local++;return;
      }
      const double w=cellweightflag ? particles(i).weight : 1.;
      if(!Kokkos::isfinite(w) || !(w>0.) ||
         !Kokkos::isfinite(species_mass(s)) || !(species_mass(s)>0.) ||
         !Kokkos::isfinite(particles(i).v[0]) ||
         !Kokkos::isfinite(conditional_vx(i))) local++;
    },bad);
  return bad;
}
template<class ExecSpace,class ParticleView,class SpeciesMassView,
         class ProbabilityView,class TargetView,class AfterVelocityView,
         class ImpulseView>
void checked_onepart_axial_partial_exchange(
    const ParticleView& particles,const SpeciesMassView& species_mass,
    const ProbabilityView& probability,const TargetView& target,
    const AfterVelocityView& conditional_vx,const ImpulseView& impulses,
    int nlocal,double fnum,bool cellweightflag);
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
// Reject invalid events before any particle or grain momentum is changed.
template<class ExecSpace,class ParticleView,class SpeciesMassView,
         class ProbabilityView,class TargetView,class AfterVelocityView,
         class ImpulseView>
void checked_onepart_axial_partial_exchange(
    const ParticleView& particles,const SpeciesMassView& species_mass,
    const ProbabilityView& probability,const TargetView& target,
    const AfterVelocityView& conditional_vx,const ImpulseView& impulses,
    int nlocal,double fnum,bool cellweightflag) {
  if(!std::isfinite(fnum) || !(fnum>0.) ||
     count_invalid_onepart_events<ExecSpace>(particles,species_mass,probability,
         target,conditional_vx,impulses,nlocal,cellweightflag))
    throw std::invalid_argument("invalid SPARTA gas/grain event data");
  onepart_axial_partial_exchange<ExecSpace>(particles,species_mass,probability,
      target,conditional_vx,impulses,nlocal,fnum,cellweightflag);
}
}} // namespaces
#endif
