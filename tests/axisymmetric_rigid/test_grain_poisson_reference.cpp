#include "grain_poisson_reference.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace SPARTA_NS::GrainPoissonReference;
struct ConstUniform { double operator()(){return std::exp(-0.25);} };
int main() {
 if(probability_at_least_one(0.)!=0.) return 1;
 if(std::abs(probability_at_least_one(1.e-12)/1.e-12-1.)>1.e-12) return 2;
 if(std::abs(probability_at_least_one(1.)-(1.-std::exp(-1.)))>1.e-14) return 3;
 auto rng=ConstUniform();
 if(sample_waiting_time(0.,rng)!=0) return 4;
 if(sample_waiting_time(0.6,rng)!=2) return 5;
 try {probability_at_least_one(-1.);return 6;}
 catch(const std::invalid_argument&) {}
 std::cout<<"PASS Poisson collision-count reference\n";
}
