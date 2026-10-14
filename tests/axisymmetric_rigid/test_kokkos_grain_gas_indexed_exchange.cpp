#include "kokkos_grain_gas_indexed_exchange.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);
 int error=0;
 {
  using Ex=Kokkos::DefaultExecutionSpace;
  using Mem=typename Ex::memory_space;
  SPARTA_NS::AxisymmetricGrains::GasIndexedExchange<Ex> events(3);
  Kokkos::View<double*[2],Mem> v("v",3), impulse("impulse",2);
  auto av=Kokkos::create_mirror_view(v);
  auto a=Kokkos::create_mirror_view(events.active);
  auto g=Kokkos::create_mirror_view(events.grain);
  auto after=Kokkos::create_mirror_view(events.after);
  auto mass=Kokkos::create_mirror_view(events.molecular_mass);
  auto w=Kokkos::create_mirror_view(events.weight);
  for(int i=0;i<3;i++) {
   av(i,0)=10.+i; av(i,1)=0.;
   a(i)=(i!=1); g(i)=0; mass(i)=2.;w(i)=3.;
   after(i,0)=8.+i;after(i,1)=1.;
  }
  g(2)=1;
  Kokkos::deep_copy(v,av); Kokkos::deep_copy(events.active,a);
  Kokkos::deep_copy(events.grain,g);
  Kokkos::deep_copy(events.after,after);
  Kokkos::deep_copy(events.molecular_mass,mass);
  Kokkos::deep_copy(events.weight,w);
  Kokkos::deep_copy(impulse,0.);
  SPARTA_NS::AxisymmetricGrains::gas_indexed_momentum_exchange(events,v,impulse);
  auto p=Kokkos::create_mirror_view(impulse);
  Kokkos::deep_copy(p,impulse);Kokkos::deep_copy(av,v);
  // Each selected gas particle loses axial momentum 2*3*(2)=12,
  // and gains radial momentum 2*3*(1)=6.
  if(std::abs(p(0,0)-12.)>1.e-12 ||
     std::abs(p(1,0)-12.)>1.e-12 ||
     std::abs(p(0,1)+6.)>1.e-12 ||
     std::abs(p(1,1)+6.)>1.e-12 ||
     std::abs(av(1,0)-11.)>1.e-12 ||
     std::abs(av(1,1))>1.e-12) error=1;
 }
 Kokkos::finalize();
 if(error) {std::cerr<<"FAIL gas-indexed momentum exchange\n";return 1;}
 std::cout<<"PASS race-free gas-indexed exchange bookkeeping\n";
}
