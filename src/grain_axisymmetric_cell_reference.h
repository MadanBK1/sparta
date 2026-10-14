/* Axisymmetric annular control-volume geometry for DSMC diagnostics.
   Coordinates: x axial, r radial >= 0.
   A rectangular meridional cell [x0,x1] x [r0,r1] represents
   V = pi*(r1^2-r0^2)*(x1-x0) in 3D.
   This helper does NOT provide a grain collision probability or spatial
   locality of an off-axis sphere: its purpose is robust cell weighting.
*/
#ifndef SPARTA_GRAIN_AXISYMMETRIC_CELL_REFERENCE_H
#define SPARTA_GRAIN_AXISYMMETRIC_CELL_REFERENCE_H
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS { namespace GrainAxisymmetricCellReference {
inline double annular_volume(double x0,double x1,double r0,double r1) {
 if(!std::isfinite(x0)||!std::isfinite(x1)||!std::isfinite(r0)||
    !std::isfinite(r1)||!(x1>x0)||r0<0.||!(r1>r0))
   throw std::invalid_argument("invalid axisymmetric cell bounds");
 const double pi=std::acos(-1.);
 const double v=pi*(r1-r0)*(r1+r0)*(x1-x0);
 if(!std::isfinite(v) || !(v>0.))
   throw std::overflow_error("annular cell volume overflow");
 return v;
}
inline double molecular_number_density(double physical_molecule_count,
                                        double cell_volume) {
 if(!std::isfinite(physical_molecule_count)||physical_molecule_count<0.||
    !std::isfinite(cell_volume)||!(cell_volume>0.))
   throw std::invalid_argument("invalid physical count or cell volume");
 return physical_molecule_count/cell_volume;
}
}} // namespaces
#endif
