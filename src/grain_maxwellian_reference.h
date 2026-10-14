/* Analytical free-molecular encounter rate for an isolated spherical grain
   in a locally homogeneous Maxwellian gas. The rate is an expectation,
   not an event sampler and not a substitute for an axisymmetric coupling
   model. This accounts for thermal motion missing from monoenergetic beams.

   f(v) proportional to exp(-|v-U|^2/(2*sigma^2)),
   sigma = sqrt(kB*T/m), s = |U-Vgrain|.
   E[|v-Vgrain|] = sigma*sqrt(2/pi)*exp(-s^2/(2*sigma^2))
                   +(s+sigma^2/s)*erf(s/(sqrt(2)*sigma)).
   At s=0 the limit is sqrt(8/pi)*sigma.
*/
#ifndef SPARTA_GRAIN_MAXWELLIAN_REFERENCE_H
#define SPARTA_GRAIN_MAXWELLIAN_REFERENCE_H
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS { namespace GrainMaxwellianReference {
inline double mean_relative_speed(double drift_speed,double thermal_sigma) {
  if (!std::isfinite(drift_speed) || drift_speed<0. ||
      !std::isfinite(thermal_sigma) || !(thermal_sigma>0.))
    throw std::invalid_argument("invalid Maxwellian speed parameters");
  const double pi=std::acos(-1.0);
  const double a=drift_speed/thermal_sigma;
  if(a<1.e-4) {
    // Expansion through order a^4 to avoid catastrophic cancellation.
    const double t=a*a;
    return std::sqrt(8.0/pi)*thermal_sigma*
           (1.0+t/6.0-t*t/120.0);
  }
  const double e=std::exp(-0.5*a*a);
  return thermal_sigma*(std::sqrt(2.0/pi)*e+
           (a+1.0/a)*std::erf(a/std::sqrt(2.0)));
}
inline double expected_impacts(double number_density,double radius,
        double drift_speed,double thermal_sigma,double dt) {
  if (!std::isfinite(number_density) || number_density<0. ||
      !std::isfinite(radius) || !(radius>0.) ||
      !std::isfinite(dt) || !(dt>0.))
    throw std::invalid_argument("invalid density radius or timestep");
  const double pi=std::acos(-1.0);
  return number_density*pi*radius*radius*
         mean_relative_speed(drift_speed,thermal_sigma)*dt;
}
}} // namespaces
#endif
