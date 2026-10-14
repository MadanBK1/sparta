#include "axisymmetric_moving_axial_plane.h"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace SPARTA_NS::AxisymmetricMovingAxialPlane;
int main() {
  auto a=intersect(0.0,0.1,10.,0.,0.,0.5,2.,0.1);
  assert(a.hit && std::abs(a.time-0.0625)<1e-13);
  assert(std::abs(a.axial_position-0.625)<1e-13);
  assert(std::abs(a.radial_position-0.1)<1e-13);
  auto b=intersect(0.0,0.1,10.,3.,4.,0.5,2.,0.1);
  assert(b.hit && std::abs(b.radial_position-
    std::hypot(0.1+3.*0.0625,4.*0.0625))<1e-13);
  assert(!intersect(0.0,0.1,10.,0.,0.,0.5,10.,0.1).hit);
  assert(!intersect(0.0,0.1,10.,0.,0.,0.5,2.,0.01).hit);
  assert(!intersect(0.6,0.1,10.,0.,0.,0.5,2.,0.1).hit);
  // A bounded axisymmetric axial face is an annulus with inner/outer radii.
  assert(intersect_annular_face(0.,0.1,10.,0.,0.,0.5,2.,
                               0.05,0.15,0.1).hit);
  assert(!intersect_annular_face(0.,0.1,10.,0.,0.,0.5,2.,
                                0.11,0.15,0.1).hit);
  assert(!intersect_annular_face(0.,0.1,10.,0.,0.,0.5,2.,
                                0.0,0.09,0.1).hit);
  assert(!intersect_annular_face(0.,0.1,10.,0.,0.,0.5,2.,
                                0.15,0.05,0.1).hit);
  std::cout << "PASS: axial moving-plane collision times and cylindrical radial propagation\n";
}
