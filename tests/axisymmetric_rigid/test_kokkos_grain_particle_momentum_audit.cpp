#include "kokkos_grain_particle_momentum_audit.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
struct MockPart {
 int ispecies;
 double weight;
 double v[3];
};
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);
 int failed=0;
 {
  using Ex=Kokkos::DefaultExecutionSpace;
  using Mem=typename Ex::memory_space;
  Kokkos::View<MockPart*,Mem> particles("particles",2);
  Kokkos::View<double*,Mem> mass("mass",2);
  auto h=Kokkos::create_mirror_view(particles);
  auto hm=Kokkos::create_mirror_view(mass);
  h(0).ispecies=0;h(0).weight=2.;h(0).v[0]=5.;
  h(1).ispecies=1;h(1).weight=3.;h(1).v[0]=-1.;
  hm(0)=2.;hm(1)=4.;
  Kokkos::deep_copy(particles,h);Kokkos::deep_copy(mass,hm);
  using SPARTA_NS::AxisymmetricGrains::axial_gas_momentum;
  const double p=axial_gas_momentum<Ex>(particles,mass,2,10.,true);
  const double q=axial_gas_momentum<Ex>(particles,mass,2,10.,false);
  if(std::abs(p-80.)>1.e-12 || std::abs(q-60.)>1.e-12) failed=1;
 }
 Kokkos::finalize();
 if(failed){std::cerr<<"FAIL momentum view adapter\n";return 1;}
 std::cout<<"PASS momentum view adapter\n";
}
