/* Kokkos paired momentum update for externally validated collision events.
   Does not generate/select events or model spherical-grain cross sections.
   One event per distinct gas index; no repeated gas indices allowed.
   Each event changes velocity of its DSMC representative and accumulates
   the opposite PHYSICAL momentum onto its assigned grain.
   All physical gas weights must be positive and supplied per event.
   A common bulk-gas axisymmetric interpretation is only valid after
   the associated statistical collision model is validated separately.
*/
#ifndef SPARTA_KOKKOS_GRAIN_PAIRED_COLLISION_H
#define SPARTA_KOKKOS_GRAIN_PAIRED_COLLISION_H
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <cmath>
namespace SPARTA_NS {
namespace AxisymmetricGrains {
template<class ExecSpace>
struct PairedCollisionEvents {
  using Mem=typename ExecSpace::memory_space;
  Kokkos::View<int*,Mem> gas_index, grain_index;
  Kokkos::View<double*[2],Mem> new_gas_velocity;
  Kokkos::View<double*,Mem> gas_molecular_mass, physical_weight;
  explicit PairedCollisionEvents(int n) :
    gas_index("event_gas",n),grain_index("event_grain",n),
    new_gas_velocity("event_gas_velocity_after",n),
    gas_molecular_mass("event_molecular_mass",n),
    physical_weight("event_physical_weight",n) {}
};
template<class ExecSpace, class GasVelocityView, class GrainImpulseView>
void apply_paired_collisions(const PairedCollisionEvents<ExecSpace>& events,
                             const GasVelocityView& gas_velocity,
                             const GrainImpulseView& grain_impulse) {
  const int n=static_cast<int>(events.gas_index.extent(0));
  auto gas=events.gas_index, grain=events.grain_index,
    after=events.new_gas_velocity, mass=events.gas_molecular_mass,
    weight=events.physical_weight;
  // Caller must validate index ranges, finite values and uniqueness of gas
  // indices BEFORE invoking this asynchronous GPU kernel.
  Kokkos::parallel_for("grain_paired_collision",
    Kokkos::RangePolicy<ExecSpace>(0,n),KOKKOS_LAMBDA(int k) {
      const int i=gas(k), j=grain(k);
      const double wmass=mass(k)*weight(k);
      const double dpx=wmass*(after(k,0)-gas_velocity(i,0));
      const double dpr=wmass*(after(k,1)-gas_velocity(i,1));
      gas_velocity(i,0)=after(k,0);
      gas_velocity(i,1)=after(k,1);
      Kokkos::atomic_add(&grain_impulse(j,0),-dpx);
      Kokkos::atomic_add(&grain_impulse(j,1),-dpr);
    });
}
} // AxisymmetricGrains
} // SPARTA_NS
#endif
