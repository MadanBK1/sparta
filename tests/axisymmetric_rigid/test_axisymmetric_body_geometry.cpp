// Run: g++ -std=c++11 -Wall -Wextra -Werror -Isrc
//   tests/axisymmetric_rigid/test_axisymmetric_body_geometry.cpp -o /tmp/test_axisrigid
//   /tmp/test_axisrigid
#include "axisymmetric_body_geometry.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
using namespace SPARTA_NS::AxisymmetricBodyGeometry;
int main() {
  const double pi=std::acos(-1.0), rho=2600., R=0.003;
  const double L=0.01;
  // Rectangle revolved about x becomes a solid cylinder.
  std::vector<Point> cylinder={{0,0},{L,0},{L,R},{0,R}};
  auto a=compute(cylinder,rho);
  double mass=rho*pi*R*R*L;
  assert(std::abs(a.mass/mass-1)<1e-12);
  assert(std::abs(a.xcm-L/2)<1e-12);
  assert(std::abs(a.inertia_axial/(mass*R*R/2)-1)<1e-12);
  assert(std::abs(a.inertia_transverse/(mass*(3*R*R+L*L)/12)-1)<1e-12);
  // Annular rectangular profile represents an annular cylinder.
  const double Ri=0.002, Ro=0.005;
  std::vector<Point> annulus={{0,Ri},{L,Ri},{L,Ro},{0,Ro}};
  auto b=compute(annulus,rho);
  mass=rho*pi*(Ro*Ro-Ri*Ri)*L;
  assert(std::abs(b.mass/mass-1)<1e-12);
  assert(std::abs(b.inertia_axial/(mass*(Ri*Ri+Ro*Ro)/2)-1)<1e-12);
  std::vector<Point> reversed(annulus.rbegin(),annulus.rend());
  auto c=compute(reversed,rho);
  assert(std::abs(c.mass/b.mass-1)<1e-12);
  std::cout << "PASS: cylinder, annulus, orientation, principal inertia\n";
}
