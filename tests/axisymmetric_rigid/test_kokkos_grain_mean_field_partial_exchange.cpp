#include "kokkos_grain_mean_field_partial_exchange.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);
 int fail=0;
 {
  using Ex=Kokkos::DefaultExecutionSpace;
  using Mem=typename Ex::memory_space;
  SPARTA_NS::AxisymmetricGrains::MeanFieldPartialEvents<Ex> e(2);
  Kokkos::View<double*[2],Mem> gas("gas",2), impulse("impulse",1);
  auto v=Kokkos::create_mirror_view(gas);
  auto t=Kokkos::create_mirror_view(e.target);
  auto p=Kokkos::create_mirror_view(e.probability);
  auto m=Kokkos::create_mirror_view(e.molecular_mass);
  auto w=Kokkos::create_mirror_view(e.weight);
  auto post=Kokkos::create_mirror_view(e.conditional_after);
  v(0,0)=10.;v(0,1)=2.; v(1,0)=4.;v(1,1)=3.;
  t(0)=0;t(1)=0;p(0)=0.25;p(1)=0.;
  m(0)=m(1)=2.;w(0)=w(1)=5.;
  post(0,0)=6.;post(0,1)=0.;post(1,0)=-9.;post(1,1)=-9.;
  Kokkos::deep_copy(gas,v);Kokkos::deep_copy(e.target,t);
  Kokkos::deep_copy(e.probability,p);Kokkos::deep_copy(e.molecular_mass,m);
  Kokkos::deep_copy(e.weight,w);Kokkos::deep_copy(e.conditional_after,post);
  Kokkos::deep_copy(impulse,0.);
  SPARTA_NS::AxisymmetricGrains::apply_mean_field_partial_exchange(e,gas,impulse);
  Kokkos::deep_copy(v,gas);
  auto j=Kokkos::create_mirror_view(impulse);
  Kokkos::deep_copy(j,impulse);
  if(std::abs(v(0,0)-9.)>1.e-12 || std::abs(v(0,1)-1.5)>1.e-12 ||
     std::abs(v(1,0)-4.)>1.e-12 || std::abs(v(1,1)-3.)>1.e-12 ||
     std::abs(j(0,0)-10.)>1.e-12 || std::abs(j(0,1)-5.)>1.e-12) fail=1;
  // Gas receives (-10,-5) physical momentum; grain receives (+10,+5).
 }
 Kokkos::finalize();
 if(fail){std::cerr<<"FAIL mean-field partial exchange\n";return 1;}
 std::cout<<"PASS mean-field partial gas-grain momentum exchange\n";
}
