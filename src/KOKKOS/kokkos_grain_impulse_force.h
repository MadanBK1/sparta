/* Convert validated per-grain momentum impulses into forces on device.
   Call after zeroing impulses and completing all gas collision events.
   No gas collision selection or back reaction is provided here.
*/
#ifndef SPARTA_KOKKOS_GRAIN_IMPULSE_FORCE_H
#define SPARTA_KOKKOS_GRAIN_IMPULSE_FORCE_H
#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS {
namespace AxisymmetricGrains {
template<class ExecSpace,class ImpulseView,class ForceView>
void add_impulse_forces(const ImpulseView& impulse, const ForceView& force,
                        int n, double dt) {
  if (!(dt>0.0) || !std::isfinite(dt) || n<0 ||
      static_cast<size_t>(n)>impulse.extent(0) ||
      static_cast<size_t>(n)>force.extent(0))
    throw std::invalid_argument("invalid impulse-to-force dimensions or timestep");
  const double inverse_dt=1.0/dt;
  Kokkos::parallel_for("grain_impulse_to_force",
    Kokkos::RangePolicy<ExecSpace>(0,n),KOKKOS_LAMBDA(const int i) {
      // One thread per grain; avoids atomics on the final force field.
      force(i,0)+=impulse(i,0)*inverse_dt;
      force(i,1)+=impulse(i,1)*inverse_dt;
    });
}
} // AxisymmetricGrains
} // SPARTA_NS
#endif
