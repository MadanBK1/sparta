/* Axial momentum-budget diagnostic for a closed gas/grain subsystem.
   Gas before and after must be computed from matching SPARTA particle
   weights/species masses; grain impulse is the NET axial impulse.
   External gas flux, body/wall reactions or active source terms invalidate
   the closed-system assertion and must be budgeted separately.
*/
#ifndef SPARTA_GRAIN_MOMENTUM_BUDGET_H
#define SPARTA_GRAIN_MOMENTUM_BUDGET_H
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace SPARTA_NS { namespace GrainMomentumBudget {
struct Balance {double residual,scale,relative;};
inline Balance axial(double gas_before,double gas_after,
                     double grain_impulse,double external_impulse=0.) {
 if(!std::isfinite(gas_before)||!std::isfinite(gas_after)||
    !std::isfinite(grain_impulse)||!std::isfinite(external_impulse))
   throw std::invalid_argument("nonfinite momentum-budget input");
 const double change=gas_after-gas_before;
 const double residual=change+grain_impulse-external_impulse;
 const double scale=std::max(1.,std::max(std::abs(change),
                      std::max(std::abs(grain_impulse),std::abs(external_impulse))));
 return {residual,scale,std::abs(residual)/scale};
}
}} // namespaces
#endif
