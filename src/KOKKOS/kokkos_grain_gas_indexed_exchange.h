/* Race-free Kokkos gas/grain momentum exchange, indexed by gas particle.
   Each gas-particle index is visited exactly once; no event compaction and
   no duplicate-index host validation are needed. Active(k) is 0 or 1;
   grain(k) identifies the recipient. Caller must validate all metadata
   and provide physically justified post-collision velocities/weights.
   IMPORTANT: This is a bookkeeping kernel, NOT a collision sampler.
   Axisymmetric radial impulse balance requires a separate model.
*/
#ifndef SPARTA_KOKKOS_GRAIN_GAS_INDEXED_EXCHANGE_H
#define SPARTA_KOKKOS_GRAIN_GAS_INDEXED_EXCHANGE_H
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <cmath>
namespace SPARTA_NS { namespace AxisymmetricGrains {
template<class ExecSpace>
struct GasIndexedExchange {
  using Mem=typename ExecSpace::memory_space;
  Kokkos::View<int*,Mem> active, grain;
  Kokkos::View<double*[2],Mem> after;
  Kokkos::View<double*,Mem> molecular_mass, weight;
  explicit GasIndexedExchange(int gas_count):
    active("gas_grain_active",gas_count), grain("gas_grain_target",gas_count),
    after("gas_velocity_after",gas_count),
    molecular_mass("gas_molecular_mass",gas_count),
    weight("gas_physical_weight",gas_count) {}
};
template<class ExecSpace,class GasVelocityView,class GrainImpulseView>
void gas_indexed_momentum_exchange(const GasIndexedExchange<ExecSpace>& events,
                                   const GasVelocityView& velocity,
                                   const GrainImpulseView& grain_impulse) {
  const int n=static_cast<int>(events.active.extent(0));
  if (velocity.extent(0)!=static_cast<size_t>(n) ||
      velocity.extent(1)<2 || grain_impulse.extent(1)<2 ||
      events.grain.extent(0)!=static_cast<size_t>(n) ||
      events.after.extent(0)!=static_cast<size_t>(n) ||
      events.after.extent(1)!=2 ||
      events.molecular_mass.extent(0)!=static_cast<size_t>(n) ||
      events.weight.extent(0)!=static_cast<size_t>(n))
    throw std::invalid_argument("gas-indexed event extent mismatch");
  auto active=events.active, target=events.grain, after=events.after;
  auto mass=events.molecular_mass, weight=events.weight;
  Kokkos::parallel_for("gas_indexed_grain_momentum_exchange",
    Kokkos::RangePolicy<ExecSpace>(0,n),KOKKOS_LAMBDA(int i) {
      if(!active(i)) return;
      const double effective_mass=mass(i)*weight(i);
      const double d0=effective_mass*(after(i,0)-velocity(i,0));
      const double d1=effective_mass*(after(i,1)-velocity(i,1));
      velocity(i,0)=after(i,0);
      velocity(i,1)=after(i,1);
      Kokkos::atomic_add(&grain_impulse(target(i),0),-d0);
      Kokkos::atomic_add(&grain_impulse(target(i),1),-d1);
    });
}
}} // namespaces
#endif
