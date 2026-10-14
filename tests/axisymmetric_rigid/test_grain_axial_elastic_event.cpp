#include "grain_axial_elastic_event.h"
#include <cmath>
#include <iostream>
using namespace SPARTA_NS::GrainAxialElasticEvent;
int main() {
 const double mg=2., mb=8., vg=10., vb=-1.;
 auto r=collide(vg,vb,0.5,4.,mb);
 const double p0=mg*vg+mb*vb,p1=mg*r.gas_velocity+mb*r.grain_velocity;
 const double e0=0.5*mg*vg*vg+0.5*mb*vb*vb;
 const double e1=0.5*mg*r.gas_velocity*r.gas_velocity+
                 0.5*mb*r.grain_velocity*r.grain_velocity;
 if(std::abs(p1-p0)>1e-12 || std::abs(e1-e0)>1e-12 ||
    std::abs(mb*(r.grain_velocity-vb)-r.impulse_to_grain)>1e-12) return 1;
 std::cout<<"PASS axial elastic collision impulse, energy and momentum\n";
}
