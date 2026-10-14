/* Gas/grain axial momentum budget from SPARTA-style device particle views.
   Computes before/after totals without copying the full arrays to host.
   Particle ID/species/weight must match between snapshots; this is a
   diagnostic for collision-only stages, not migration or resampling.
   Off-axis sphere physics is NOT validated by a balanced momentum budget.
*/
#ifndef SPARTA_KOKKOS_GRAIN_MOMENTUM_BUDGET_AUDIT_H
#define SPARTA_KOKKOS_GRAIN_MOMENTUM_BUDGET_AUDIT_H
#include "kokkos_grain_particle_momentum_audit.h"
#include <cmath>
namespace SPARTA_NS { namespace AxisymmetricGrains {
struct MomentumBudgetAudit {
  double gas_before,gas_after,grain_impulse,external_impulse,residual;
};
template<class ExecSpace,class ParticleView,class SpeciesMassView,class ImpulseView>
MomentumBudgetAudit audit_gas_grain_axial(
    const ParticleView& before,const ParticleView& after,
    const SpeciesMassView& species_mass,const ImpulseView& grain_impulse,
    int nlocal,double fnum,bool cellweightflag,double external_impulse=0.) {
  if (!std::isfinite(external_impulse) ||
      grain_impulse.extent(1)<2 ||
      before.extent(0)<static_cast<size_t>(nlocal>0?nlocal:0) ||
      after.extent(0)<static_cast<size_t>(nlocal>0?nlocal:0))
    throw std::invalid_argument("invalid axial budget views");
  const double p0=axial_gas_momentum<ExecSpace>(
    before,species_mass,nlocal,fnum,cellweightflag);
  const double p1=axial_gas_momentum<ExecSpace>(
    after,species_mass,nlocal,fnum,cellweightflag);
  double sum=0.0;
  Kokkos::parallel_reduce("grain_axial_impulse_sum",
     Kokkos::RangePolicy<ExecSpace>(0,grain_impulse.extent(0)),
     KOKKOS_LAMBDA(const int i,double& local) {
       local+=grain_impulse(i,0);
     },sum);
  return {p0,p1,sum,external_impulse,p1-p0+sum-external_impulse};
}
}} // namespaces
#endif
