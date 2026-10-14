#include "grain_beam_reference.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
 using namespace SPARTA_NS::GrainBeamReference;
 const double n=1.e18,r=75.e-6,speed=1000.,dt=1.e-8;
 const double expected=n*std::acos(-1.)*r*r*speed*dt;
 if(std::abs(expected_impacts(n,r,speed,dt)/expected-1.)>1.e-14) return 1;
 if(std::abs(expected_simulator_events(n,r,speed,dt,100.)/expected-0.01)>1.e-14) return 2;
 if(expected_impacts(n,r,0.,dt)!=0.) return 3;
 if(std::abs(expected_impacts(n,2*r,speed,dt)/expected-4.)>1.e-14) return 4;
 try {expected_impacts(-1.,r,speed,dt);return 5;}
 catch(const std::invalid_argument&) {}
 std::cout<<"PASS isolated spherical grain beam impact-rate reference\n";
}
