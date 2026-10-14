#include "grain_axisymmetric_cell_reference.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace SPARTA_NS::GrainAxisymmetricCellReference;
int main() {
 const double pi=std::acos(-1.);
 const double inner=annular_volume(0.,2.,0.,1.);
 const double outer=annular_volume(0.,2.,1.,2.);
 if(std::abs(inner-2.*pi)>1.e-12) return 1;
 if(std::abs(outer-6.*pi)>1.e-12) return 2;
 if(std::abs((inner+outer)-8.*pi)>1.e-12) return 3;
 if(std::abs(molecular_number_density(8.*pi*1.e18,inner+outer)/1.e18-1.)>1.e-12)
   return 4;
 try{annular_volume(0.,1.,-1.,1.);return 5;}
 catch(const std::invalid_argument&) {}
 std::cout<<"PASS axisymmetric annular cell volume and density\n";
}
