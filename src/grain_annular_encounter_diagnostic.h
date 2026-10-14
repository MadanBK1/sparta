/* Dilute, well-mixed annular-cell sphere encounter diagnostic.
   The annular gas field is azimuthally averaged; an individual off-axis
   sphere has no resolved 3D wake. Under uniform-in-cell mixing and
   free-molecular assumptions, a physical molecule's expected exposure
   is sigma*|v-vgrain|*dt/Vcell (small time steps only).
   A DSMC simulation particle represents W molecules. W*exposure is an
   expected PHYSICAL collision count, NOT an all-or-nothing probability
   of redirecting the full DSMC representative. When W*exposure is
   non-negligible, a physical collision sampler must represent partial
   interactions (e.g. splitting/variance-controlled estimator).
   This is deliberately diagnostic-only and has no gas velocity update.
*/
#ifndef SPARTA_GRAIN_ANNULAR_ENCOUNTER_DIAGNOSTIC_H
#define SPARTA_GRAIN_ANNULAR_ENCOUNTER_DIAGNOSTIC_H
#include "grain_axisymmetric_cell_reference.h"
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS { namespace GrainAnnularEncounterDiagnostic {
struct Result {
 double optical_depth;
 double single_molecule_probability;
 double expected_physical_impacts;
};
inline Result evaluate(double radius,double relative_speed,double dt,
                       double annular_volume,double representative_weight) {
 if(!std::isfinite(radius)||!(radius>0.)||
    !std::isfinite(relative_speed)||relative_speed<0.||
    !std::isfinite(dt)||!(dt>0.)||
    !std::isfinite(annular_volume)||!(annular_volume>0.)||
    !std::isfinite(representative_weight)||!(representative_weight>0.))
   throw std::invalid_argument("invalid annular encounter parameters");
 const double pi=std::acos(-1.);
 const double tau=pi*radius*radius*relative_speed*dt/annular_volume;
 if(!std::isfinite(tau)) throw std::overflow_error("encounter optical depth overflow");
 const double p=-std::expm1(-tau);
 const double expected=representative_weight*p;
 if(!std::isfinite(expected)) throw std::overflow_error("expected collision overflow");
 return {tau,p,expected};
}
}} // namespaces
#endif
