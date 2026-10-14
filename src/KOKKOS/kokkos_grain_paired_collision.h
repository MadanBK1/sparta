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
#include <vector>
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
// Debug/setup validation for paired events. Copies metadata to host and
// verifies one event per gas particle to prevent parallel write races.
// Not intended for every production timestep; a device-side unique-event
// generator must enforce the same invariant without host transfers.
template<class ExecSpace, class GasVelocityView, class GrainImpulseView>
void validate_paired_collisions(const PairedCollisionEvents<ExecSpace>& events,
                                const GasVelocityView& gas_velocity,
                                const GrainImpulseView& grain_impulse) {
  auto ids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),events.gas_index);
  auto gids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),events.grain_index);
  auto mass=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),events.gas_molecular_mass);
  auto weight=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),events.physical_weight);
  auto after=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),events.new_gas_velocity);
  const size_t n=ids.extent(0);
  if (gids.extent(0)!=n || mass.extent(0)!=n ||
      weight.extent(0)!=n || after.extent(0)!=n ||
      after.extent(1)!=2 || gas_velocity.extent(1)<2 ||
      grain_impulse.extent(1)<2)
    throw std::invalid_argument("inconsistent collision-event view extents");
  std::vector<unsigned char> seen(gas_velocity.extent(0),0);
  for(size_t k=0;k<n;k++) {
    const int i=ids(k), j=gids(k);
    if(i<0 || static_cast<size_t>(i)>=gas_velocity.extent(0) ||
       j<0 || static_cast<size_t>(j)>=grain_impulse.extent(0))
      throw std::invalid_argument("collision index out of bounds");
    if(seen[i]) throw std::invalid_argument("duplicate gas collision index");
    seen[i]=1;
    if(!(mass(k)>0.0) || !(weight(k)>0.0) ||
       !std::isfinite(mass(k)) || !std::isfinite(weight(k)) ||
       !std::isfinite(after(k,0)) || !std::isfinite(after(k,1)))
      throw std::invalid_argument("invalid collision parameters");
  }
}
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
