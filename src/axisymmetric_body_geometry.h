/* Axisymmetric mass properties for a solid of revolution.
   SPARTA coordinates: x=axial, y=radial, y>=0.
   Standalone implementation; not integrated into fix rigid yet.
*/
#ifndef SPARTA_AXISYMMETRIC_BODY_GEOMETRY_H
#define SPARTA_AXISYMMETRIC_BODY_GEOMETRY_H
#include <cmath>
#include <stdexcept>
#include <vector>
namespace SPARTA_NS {
namespace AxisymmetricBodyGeometry {
struct Point { double x,y; };
struct Properties {
  double volume, mass, xcm;
  double inertia_axial, inertia_transverse;
};
inline Properties compute(const std::vector<Point>& polygon, double density)
{
  if (polygon.size() < 3 || !(density > 0.0))
    throw std::invalid_argument("invalid axisymmetric polygon or density");
  // Signed fan triangulation with degree-3 exact quadrature.
  const double b[4][3] = {{1./3,1./3,1./3},
    {0.6,0.2,0.2},{0.2,0.6,0.2},{0.2,0.2,0.6}};
  const double w[4] = {-27./48,25./48,25./48,25./48};
  double r1=0, xr=0, x2r=0, r3=0, signed_area=0;
  for (size_t i=0; i<polygon.size(); ++i) {
    Point a=polygon[i], c=polygon[(i+1)%polygon.size()];
    if (a.y < 0.0 || c.y < 0.0)
      throw std::invalid_argument("negative radius in axisymmetric polygon");
    double det=a.x*c.y-c.x*a.y;
    signed_area += 0.5*det;
    for (int q=0;q<4;q++) {
      double x=b[q][1]*a.x+b[q][2]*c.x;
      double r=b[q][1]*a.y+b[q][2]*c.y;
      double scale=0.5*det*w[q];
      r1 += scale*r;
      xr += scale*x*r;
      x2r += scale*x*x*r;
      r3 += scale*r*r*r;
    }
  }
  if (!(std::abs(r1)>0.0) || !(std::abs(signed_area)>0.0))
    throw std::invalid_argument("zero-volume axisymmetric polygon");
  // Winding of a closed polygon can be clockwise or counterclockwise.
  double sign = r1<0 ? -1.0:1.0;
  r1 *= sign; xr *= sign; x2r *= sign; r3 *= sign;
  const double pi=std::acos(-1.0);
  double volume=2*pi*r1;
  double xcm=xr/r1;
  double ix=density*2*pi*r3;
  // About a transverse axis through COM: azimuthal mean y^2 = r^2/2.
  double it=density*2*pi*(x2r-2*xcm*xr+xcm*xcm*r1+0.5*r3);
  return {volume,density*volume,xcm,ix,it};
}
} // namespace AxisymmetricBodyGeometry
} // namespace SPARTA_NS
#endif
