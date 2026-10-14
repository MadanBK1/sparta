/* GPU-portable bookkeeping for already-sampled gas/grain events.
   IMPORTANT: This does not select physical collisions or establish an
   axisymmetric gas-grain collision rate. Events and their physical
   molecule weights must be provided by a separately validated operator.
   The net radial momentum of an axisymmetric gas population is zero:
   these 2-component bookkeeping events are a reduced-model convention,
   not a rigorous conservation theorem for 3D spherical grains.
*/
#ifndef SPARTA_KOKKOS_GRAIN_IMPULSE_EVENTS_H
#define SPARTA_KOKKOS_GRAIN_IMPULSE_EVENTS_H
#include <Kokkos_Core.hpp>
namespace SPARTA_NS {
namespace AxisymmetricGrains {
template<class ExecSpace>
struct ImpulseEvents {
  using Mem=typename ExecSpace::memory_space;
  Kokkos::View<int*,Mem> grain;
  Kokkos::View<double*[2],Mem> delta_gas_p;
  Kokkos::View<double*,Mem> physical_weight;
  explicit ImpulseEvents(int n) :
    grain("impulse_grain_id",n),
    delta_gas_p("impulse_gas_momentum_per_molecule",n),
    physical_weight("impulse_weight",n) {}
};
// Each event's delta_gas_p is momentum change PER physical molecule.
// The caller must guarantee 0<=grain(e)<n_grains, finite data and w>0.
// Accumulates solid IMPULSE rather than force; does not alter SPARTA DSMC
// particles. Conversion to body force requires a timestep and must not
// double-apply physical simulation particle weighting.
template<class ExecSpace, class GrainImpulseView>
void accumulate_grain_impulses(const ImpulseEvents<ExecSpace>& events,
                              const GrainImpulseView& solid_impulse) {
  auto index=events.grain, dp=events.delta_gas_p,
       weight=events.physical_weight;
  const int n=static_cast<int>(index.extent(0));
  Kokkos::parallel_for("grain_impulse_events",
    Kokkos::RangePolicy<ExecSpace>(0,n),KOKKOS_LAMBDA(int e) {
      int id=index(e);
      double w=weight(e);
      Kokkos::atomic_add(&solid_impulse(id,0),-w*dp(e,0));
      Kokkos::atomic_add(&solid_impulse(id,1),-w*dp(e,1));
    });
}
} // namespace AxisymmetricGrains
} // namespace SPARTA_NS
#endif
