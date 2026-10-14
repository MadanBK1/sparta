#include "axisymmetric_body_geometry.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace SPARTA_NS::AxisymmetricBodyGeometry;
int main() {
  const double pi=std::acos(-1.0), R=0.003, rho=2600.0;
  const double exactVolume=4.0*pi*R*R*R/3.0;
  const double exactI=2.0*(rho*exactVolume)*R*R/5.0;
  double prev=1.0;
  for (int n: {16,32,64,128,256}) {
    std::vector<Point> halfdisk;
    for(int j=0;j<=n;j++) {
      double theta=j*pi/n;
      halfdisk.push_back({R*std::cos(theta),R*std::sin(theta)});
    }
    const auto p=compute(halfdisk,rho);
    const double e=std::abs(p.volume/exactVolume-1.0);
    if (!(e<prev && std::abs(p.xcm)<1.e-13 &&
         std::abs(p.inertia_axial/exactI-1.0)<prev &&
         std::abs(p.inertia_transverse/exactI-1.0)<prev)) {
      std::cerr << "FAIL resolution " << n << " err " << e << "\n";
      return EXIT_FAILURE;
    }
    prev=e;
  }
  if (!(prev<1.0e-4)) return EXIT_FAILURE;
  std::cout << "PASS: polygonal sphere mass and both inertias converge to 3D analytic values\n";
}
