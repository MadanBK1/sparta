#include "grain_dsmc_mass_reference.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace SPARTA_NS::GrainDSMCMassReference;
int main() {
 const double fnum=1.e10, mass=4.65e-26;
 const double unweighted=represented_mass(fnum,mass,false,20.);
 const double weighted=represented_mass(fnum,mass,true,20.);
 if(std::abs(unweighted/fnum/mass-1.)>1.e-12) return 1;
 if(std::abs(weighted/unweighted-20.)>1.e-12) return 2;
 const double dp=axial_momentum_change(fnum,mass,true,20.,10.,7.);
 if(std::abs(dp/weighted+3.)>1.e-12) return 3;
 if(represented_mass(fnum,mass,false,0.)!=unweighted) return 4;
 try {represented_mass(fnum,mass,true,0.);return 5;}
 catch(const std::invalid_argument&) {}
 std::cout<<"PASS SPARTA DSMC representative mass audit\n";
}
