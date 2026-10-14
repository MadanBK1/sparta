/* Reference representation of partial physical-molecule collisions.
   Preserve two weighted gas populations instead of replacing them by one
   mean velocity. This is bookkeeping ONLY: it does not select collisions,
   compute a scattering law, or create particles in SPARTA.
   The collision participants are weighted W*p; nonparticipants W*(1-p).
   Their joint momentum and kinetic energy are exactly those of the
   mixture, unlike a mean-velocity replacement, which loses variance.
*/
#ifndef SPARTA_GRAIN_WEIGHTED_SPLIT_REFERENCE_H
#define SPARTA_GRAIN_WEIGHTED_SPLIT_REFERENCE_H
#include <cmath>
#include <stdexcept>
namespace SPARTA_NS { namespace GrainWeightedSplitReference {
struct Population { double weight, vx, vr; };
struct Split {
 Population uncollided, collided;
 double grain_impulse_x, grain_impulse_r;
 double mean_field_missing_kinetic_energy;
};
inline Split split(double weight,double molecule_mass,double probability,
                   double pre_x,double pre_r,double post_x,double post_r) {
 if(!std::isfinite(weight)||!(weight>0.)||
    !std::isfinite(molecule_mass)||!(molecule_mass>0.)||
    !std::isfinite(probability)||probability<0.||probability>1.||
    !std::isfinite(pre_x)||!std::isfinite(pre_r)||
    !std::isfinite(post_x)||!std::isfinite(post_r))
   throw std::invalid_argument("invalid weighted split input");
 const double participating=weight*probability;
 const double remainder=weight-participating;
 const double dx=post_x-pre_x, dr=post_r-pre_r;
 const double impulse=-molecule_mass*participating;
 const double deficit=0.5*molecule_mass*weight*probability*
                    (1.-probability)*(dx*dx+dr*dr);
 return {{remainder,pre_x,pre_r},{participating,post_x,post_r},
         impulse*dx,impulse*dr,deficit};
}
}} // namespaces
#endif
