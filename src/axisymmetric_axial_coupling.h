/* Standalone axial collision-impulse accounting for axisymmetric rigid bodies.
   The caller must apply physical DSMC particle weight exactly once.
   Not integrated into SPARTA's mover/compute-surf paths yet. */
#ifndef SPARTA_AXISYMMETRIC_AXIAL_COUPLING_H
#define SPARTA_AXISYMMETRIC_AXIAL_COUPLING_H
#include <stdexcept>
#include <cmath>
namespace SPARTA_NS {
namespace AxisymmetricAxialCoupling {
struct Ledger {
  double gas_delta_p = 0.0;
  double body_delta_p = 0.0;
  double body_mass = 0.0;
  double body_velocity = 0.0;
  explicit Ledger(double mass, double velocity=0.0) :
    body_mass(mass), body_velocity(velocity) {
    if (!(mass > 0.0) || !std::isfinite(mass) || !std::isfinite(velocity))
      throw std::invalid_argument("invalid body mass or velocity");
  }
  void collision(double molecular_mass, double particle_weight,
                 double v_before_x, double v_after_x) {
    if (!(molecular_mass > 0.0) || !(particle_weight > 0.0) ||
        !std::isfinite(molecular_mass) || !std::isfinite(particle_weight) ||
        !std::isfinite(v_before_x) || !std::isfinite(v_after_x))
      throw std::invalid_argument("invalid DSMC collision parameters");
    double dp = molecular_mass * particle_weight *
                (v_after_x - v_before_x);
    gas_delta_p += dp;
    body_delta_p -= dp;
  }
  double gas_force(double dt) const {
    if (!(dt > 0.0) || !std::isfinite(dt))
      throw std::invalid_argument("invalid timestep");
    return gas_delta_p / dt;
  }
  double body_force(double dt) const { return -gas_force(dt); }
  // Reference impulse step only. Integrated SPARTA code must retain its
  // velocity-Verlet half-kick timing rather than calling this directly.
  void advance_velocity(double dt, double external_force=0.0) {
    if (!(dt > 0.0) || !std::isfinite(dt) || !std::isfinite(external_force))
      throw std::invalid_argument("invalid timestep or external force");
    body_velocity += (body_delta_p + external_force*dt) / body_mass;
    reset();
  }
  void reset() { gas_delta_p=0.0; body_delta_p=0.0; }
};
} // namespace AxisymmetricAxialCoupling
} // namespace SPARTA_NS
#endif
