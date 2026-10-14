#include "axisymmetric_axial_coupling.h"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace SPARTA_NS::AxisymmetricAxialCoupling;
int main() {
  Ledger b(2.0);
  b.collision(0.5, 4.0, 8.0, 2.0);
  b.collision(0.5, 4.0, 3.0, 5.0);
  assert(std::abs(b.gas_delta_p+8.0)<1.e-13);
  assert(std::abs(b.body_delta_p-8.0)<1.e-13);
  assert(std::abs(b.body_delta_p+b.gas_delta_p)<1.e-13);
  assert(std::abs(b.body_force(0.25)-32.0)<1.e-13);
  b.advance_velocity(0.25, 4.0);
  assert(std::abs(b.body_velocity-4.5)<1.e-13);
  assert(b.gas_delta_p==0 && b.body_delta_p==0);
  std::cout << "PASS: axial impulse, action/reaction, force, velocity update\n";
}
