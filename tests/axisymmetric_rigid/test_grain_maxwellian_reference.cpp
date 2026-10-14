#include "grain_maxwellian_reference.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace SPARTA_NS::GrainMaxwellianReference;
int main() {
 const double pi=std::acos(-1.),sigma=250.;
 const double low=std::sqrt(8./pi)*sigma;
 if(std::abs(mean_relative_speed(0.,sigma)/low-1.)>1.e-13) return 1;
 if(std::abs(mean_relative_speed(1.e-7,sigma)/low-1.)>1.e-12) return 2;
 if(std::abs(mean_relative_speed(1.e6,sigma)/1.e6-1.)>1.e-6) return 3;
 if(!(mean_relative_speed(400.,sigma)>low)) return 4;
 const double n=1.e18,r=75.e-6,dt=1.e-8;
 const double expected=n*pi*r*r*low*dt;
 if(std::abs(expected_impacts(n,r,0.,sigma,dt)/expected-1.)>1.e-13)
   return 5;
 if(std::abs(expected_impacts(n,2*r,0.,sigma,dt)/expected-4.)>1.e-13)
   return 6;
 try {mean_relative_speed(0.,0.);return 7;}
 catch(const std::invalid_argument&) {}
 std::cout<<"PASS Maxwellian sphere encounter rate limiting cases\n";
}
