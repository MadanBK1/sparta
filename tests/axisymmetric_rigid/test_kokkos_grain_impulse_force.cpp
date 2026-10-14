#include "kokkos_grain_impulse_force.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  int fail=0;
  {
    using Ex=Kokkos::DefaultExecutionSpace;
    using Mem=typename Ex::memory_space;
    Kokkos::View<double*[2],Mem> impulse("impulse",2);
    Kokkos::View<double*[2],Mem> force("force",2);
    auto ih=Kokkos::create_mirror_view(impulse);
    auto fh=Kokkos::create_mirror_view(force);
    ih(0,0)=2.;ih(0,1)=-4.;
    ih(1,0)=0.;ih(1,1)=3.;
    fh(0,0)=1.;fh(0,1)=2.;
    fh(1,0)=3.;fh(1,1)=4.;
    Kokkos::deep_copy(impulse,ih);
    Kokkos::deep_copy(force,fh);
    SPARTA_NS::AxisymmetricGrains::add_impulse_forces<Ex>(impulse,force,2,0.5);
    Kokkos::deep_copy(fh,force);
    if(std::abs(fh(0,0)-5.)>1e-12 ||
       std::abs(fh(0,1)+6.)>1e-12 ||
       std::abs(fh(1,0)-3.)>1e-12 ||
       std::abs(fh(1,1)-10.)>1e-12) fail=1;
  }
  Kokkos::finalize();
  if(fail) {std::cerr<<"FAIL impulse-to-force\n"; return 1;}
  std::cout<<"PASS impulse-to-force\n";
}
