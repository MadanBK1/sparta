/* Device-only, read-only dilute annular encounter intensity diagnostic.
   One Kokkos thread per representative: no particle velocity updates.
   Input 'physical_weight' counts physical molecules represented by a
   SPARTA particle, NOT a collision probability for the representative.
   For an off-axis physical sphere this is a well-mixed statistical
   closure, not a resolved 3D sphere/wake.
*/
#ifndef SPARTA_KOKKOS_GRAIN_ANNULAR_ENCOUNTER_H
#define SPARTA_KOKKOS_GRAIN_ANNULAR_ENCOUNTER_H
#include <Kokkos_Core.hpp>
namespace SPARTA_NS { namespace GrainAnnularKokkos {
template<class ExecSpace, class SpeedView, class WeightView, class ResultView>
void expected_counts(const SpeedView& speed, const WeightView& physical_weight,
                     const ResultView& expected,
                     double grain_radius,double annular_volume,double dt) {
  const int n=static_cast<int>(speed.extent(0));
  const double pi=3.1415926535897932384626433832795;
  const double coefficient=pi*grain_radius*grain_radius*dt/annular_volume;
  Kokkos::parallel_for("grain_annular_expected_physical_impacts",
    Kokkos::RangePolicy<ExecSpace>(0,n),
    KOKKOS_LAMBDA(const int i) {
      const double tau=coefficient*speed(i);
      // Kokkos::exp handles portable device math; tau >= 0 is required.
      const double probability=-Kokkos::expm1(-tau);
      expected(i)=physical_weight(i)*probability;
    });
}
// Preconditions: radius > 0, volume > 0, dt > 0; speed >= 0,
// weights > 0, output extent >= input extent. Validated by caller.
// This diagnostic must never directly scatter momentum to the gas.
}} // namespace
#endif
