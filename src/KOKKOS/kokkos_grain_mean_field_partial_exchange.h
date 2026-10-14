/* Mean-field partial-participation operator for an annular DSMC grain model.
   One GPU thread owns each gas representative, so no gas write races.
   probability(i) is a physical-molecule encounter probability in [0,1],
   NOT a per-superparticle Bernoulli probability. In the conditional
   collision model, a fraction p of the W represented molecules changes
   from pre to post velocity; the uncollided fraction keeps pre velocity.
   The stored representative velocity becomes the mean of the mixture.
   Expected axial momentum is conserved against the grain impulse;
   the mixed gas kinetic energy is NOT preserved by this mean velocity.
   This requires a split/thermal-fluctuation model for full DSMC fidelity.
   Radial axisymmetric momentum must be handled with care separately.
   No physical collision-event selection is implemented here.
*/
#ifndef SPARTA_KOKKOS_GRAIN_MEAN_FIELD_PARTIAL_EXCHANGE_H
#define SPARTA_KOKKOS_GRAIN_MEAN_FIELD_PARTIAL_EXCHANGE_H
#include <Kokkos_Core.hpp>
#include <stdexcept>
namespace SPARTA_NS { namespace AxisymmetricGrains {
template<class ExecSpace>
struct MeanFieldPartialEvents {
  using Mem=typename ExecSpace::memory_space;
  Kokkos::View<int*,Mem> target;
  Kokkos::View<double*,Mem> probability, molecular_mass, weight;
  Kokkos::View<double*[2],Mem> conditional_after;
  explicit MeanFieldPartialEvents(int n):
    target("partial_grain_index",n),
    probability("partial_probability",n),
    molecular_mass("partial_molecule_mass",n),
    weight("partial_physical_weight",n),
    conditional_after("partial_postcollision_velocity",n) {}
};
template<class ExecSpace,class GasView,class ImpulseView>
void apply_mean_field_partial_exchange(
    const MeanFieldPartialEvents<ExecSpace>& events,
    const GasView& gas, const ImpulseView& grain_impulse) {
  const int n=static_cast<int>(events.target.extent(0));
  if(gas.extent(0)!=static_cast<size_t>(n) || gas.extent(1)<2 ||
     grain_impulse.extent(1)<2 ||
     events.probability.extent(0)!=static_cast<size_t>(n) ||
     events.molecular_mass.extent(0)!=static_cast<size_t>(n) ||
     events.weight.extent(0)!=static_cast<size_t>(n) ||
     events.conditional_after.extent(0)!=static_cast<size_t>(n) ||
     events.conditional_after.extent(1)!=2)
    throw std::invalid_argument("partial-exchange view extent mismatch");
  auto target=events.target, p=events.probability;
  auto mass=events.molecular_mass, w=events.weight;
  auto post=events.conditional_after;
  Kokkos::parallel_for("grain_mean_field_partial_exchange",
    Kokkos::RangePolicy<ExecSpace>(0,n),KOKKOS_LAMBDA(int i) {
      const double fraction=p(i);
      if(fraction==0.) return;
      const double mass_part=mass(i)*w(i)*fraction;
      const double dv0=post(i,0)-gas(i,0);
      const double dv1=post(i,1)-gas(i,1);
      gas(i,0)+=fraction*dv0;
      gas(i,1)+=fraction*dv1;
      Kokkos::atomic_add(&grain_impulse(target(i),0),-mass_part*dv0);
      Kokkos::atomic_add(&grain_impulse(target(i),1),-mass_part*dv1);
    });
}
}} // namespaces
#endif
