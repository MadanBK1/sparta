#include "grain_exposure_reference.h"
#include <cmath>
#include <iostream>
int main() {
 using namespace SPARTA_NS::GrainExposureReference;
 const auto a=evaluate(1.e18,75.e-6,0.,250.,1.e-8);
 const auto b=evaluate(1.e18,75.e-6,0.,250.,2.e-8);
 if(!(a.mean_impacts>0.) || std::abs(b.mean_impacts/ a.mean_impacts-2.)>1.e-12) return 1;
 if(std::abs(a.probability_impact-(-std::expm1(-a.mean_impacts)))>1.e-12) return 2;
 if(evaluate(0.,75.e-6,0.,250.,1.e-8).probability_impact!=0.) return 3;
 std::cout<<"PASS spherical grain exposure composition\n";
}
