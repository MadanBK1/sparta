#include "grain_weighted_split_reference.h"
#include <cmath>
#include <iostream>
int main() {
 using namespace SPARTA_NS::GrainWeightedSplitReference;
 const double W=1000.,m=2.,p=.2;
 const auto s=split(W,m,p,10.,3.,4.,-1.);
 if(std::abs(s.uncollided.weight-800.)>1.e-12 ||
    std::abs(s.collided.weight-200.)>1.e-12) return 1;
 const double original_px=W*m*10.;
 const double changed_px=m*(s.uncollided.weight*10.+
                              s.collided.weight*4.);
 if(std::abs(original_px-changed_px-s.grain_impulse_x)>1.e-10) return 2;
 const double before=0.5*W*m*(10.*10.+3.*3.);
 const double after=0.5*m*(s.uncollided.weight*(10.*10.+3.*3.)+
                             s.collided.weight*(4.*4.+1.));
 const double mean_x=10.+p*(4.-10.);
 const double mean_r=3.+p*(-1.-3.);
 const double mean_energy=0.5*W*m*(mean_x*mean_x+mean_r*mean_r);
 if(std::abs(after-mean_energy-s.mean_field_missing_kinetic_energy)>1.e-9)
   return 3;
 if(!(after>mean_energy) || !(before>0.)) return 4;
 const auto zero=split(W,m,0.,10.,3.,4.,-1.);
 if(zero.collided.weight!=0.||zero.mean_field_missing_kinetic_energy!=0.) return 5;
 const auto all=split(W,m,1.,10.,3.,4.,-1.);
 if(all.uncollided.weight!=0.||all.mean_field_missing_kinetic_energy!=0.) return 6;
 std::cout<<"PASS weighted partial-collision splitting and energy variance\n";
}
