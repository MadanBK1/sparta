#include "grain_momentum_budget.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
 using SPARTA_NS::GrainMomentumBudget::axial;
 auto a=axial(100.,92.,8.);
 if(a.residual!=0.||a.relative!=0.) return 1;
 auto b=axial(100.,92.,10.,2.);
 if(b.residual!=0.) return 2;
 auto c=axial(100.,92.,7.);
 if(std::abs(c.residual+1.)>1.e-12 || !(c.relative>0.)) return 3;
 try{axial(NAN,0.,0.);return 4;}
 catch(const std::invalid_argument&) {}
 std::cout<<"PASS gas-grain axial momentum budget\n";
}
