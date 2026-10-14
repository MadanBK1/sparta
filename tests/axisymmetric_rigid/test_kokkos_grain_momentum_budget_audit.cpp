#include "kokkos_grain_momentum_budget_audit.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
struct TestPart {int ispecies;double weight;double v[3];};
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);
 int fail=0;
 {
 using Ex=Kokkos::DefaultExecutionSpace;
 using Mem=typename Ex::memory_space;
 Kokkos::View<TestPart*,Mem> before("before",2),after("after",2);
 Kokkos::View<double*,Mem> mass("species_mass",1);
 Kokkos::View<double*[2],Mem> impulse("impulse",2);
 auto b=Kokkos::create_mirror_view(before);
 auto a=Kokkos::create_mirror_view(after);
 auto m=Kokkos::create_mirror_view(mass);
 auto j=Kokkos::create_mirror_view(impulse);
 m(0)=2.;
 for(int i=0;i<2;i++){
  b(i).ispecies=0; b(i).weight=1.;
  b(i).v[0]=10.;a(i)=b(i);a(i).v[0]=8.;
  j(i,0)=4.;j(i,1)=0.;
 }
 Kokkos::deep_copy(before,b);Kokkos::deep_copy(after,a);
 Kokkos::deep_copy(mass,m);Kokkos::deep_copy(impulse,j);
 auto result=SPARTA_NS::AxisymmetricGrains::audit_gas_grain_axial<Ex>(
     before,after,mass,impulse,2,1.,false);
 if(std::abs(result.residual)>1.e-12 ||
    std::abs(result.gas_before-40.)>1.e-12 ||
    std::abs(result.gas_after-32.)>1.e-12) fail=1;
 }
 Kokkos::finalize();
 if(fail){std::cerr<<"FAIL axial budget audit\n";return 1;}
 std::cout<<"PASS device-view gas-grain axial budget audit\n";
}
