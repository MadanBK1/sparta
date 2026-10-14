/* Moving plane x = xwall0 + wall_velocity*t in SPARTA x-axial,
   y-radial axisymmetric coordinates. This is a restricted, analytical
   collision-time kernel for an axial face, NOT a general axi moving line.
   No particle rotation/azimuthal velocity affects an axial plane hit.
   The caller must validate face radial bounds at the collision point.
*/
#ifndef SPARTA_AXISYMMETRIC_MOVING_AXIAL_PLANE_H
#define SPARTA_AXISYMMETRIC_MOVING_AXIAL_PLANE_H
#include <cmath>
namespace SPARTA_NS {
namespace AxisymmetricMovingAxialPlane {
struct Hit {
  bool hit;
  double time;
  double axial_position;
  double radial_position;
};
inline Hit intersect(double x0, double r0, double vx,
                     double vr, double vtheta,
                     double wall_x0, double wall_vx, double dt) {
  const Hit miss = {false,0.0,0.0,0.0};
  if (!(dt>0.0) || !std::isfinite(dt) ||
      !std::isfinite(x0) || !std::isfinite(r0) ||
      !std::isfinite(vx) || !std::isfinite(vr) ||
      !std::isfinite(vtheta) || !std::isfinite(wall_x0) ||
      !std::isfinite(wall_vx) || r0<0.0) return miss;
  const double relative_v = vx-wall_vx;
  if (relative_v==0.0) return miss;
  const double t=(wall_x0-x0)/relative_v;
  if (!(t>0.0 && t<=dt)) return miss;
  const double ry = r0+vr*t;
  const double rz = vtheta*t;
  const double r=std::hypot(ry,rz);
  return {true,t,x0+vx*t,r};
}
} // namespace AxisymmetricMovingAxialPlane
} // namespace SPARTA_NS
#endif
