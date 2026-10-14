#include "kokkos_axisymmetric_grains.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
int main(int argc,char**argv) {
  Kokkos::initialize(argc,argv);
  int status=0;
  {
    using Exec=Kokkos::DefaultExecutionSpace;
    SPARTA_NS::AxisymmetricGrains::State<Exec> g(1);
    auto p=Kokkos::create_mirror_view(g.position);
    auto v=Kokkos::create_mirror_view(g.velocity);
    auto f=Kokkos::create_mirror_view(g.force);
    auto m=Kokkos::create_mirror_view(g.mass);
    p(0,0)=0.;p(0,1)=1.;v(0,0)=0.;v(0,1)=0.;
    f(0,0)=4.;f(0,1)=0.;m(0)=2.;
    Kokkos::deep_copy(g.position,p);Kokkos::deep_copy(g.velocity,v);
    Kokkos::deep_copy(g.force,f);Kokkos::deep_copy(g.mass,m);
    SPARTA_NS::AxisymmetricGrains::kick_drift(g,0.1);
    SPARTA_NS::AxisymmetricGrains::finish_kick(g,0.1);
    Kokkos::deep_copy(p,g.position);Kokkos::deep_copy(v,g.velocity);
    if(std::abs(p(0,0)-0.01)>1e-12 ||
       std::abs(v(0,0)-0.2)>1e-12 ||
       std::abs(p(0,1)-1.)>1e-12) status=1;
    SPARTA_NS::AxisymmetricGrains::zero_forces(g);
    Kokkos::deep_copy(f,g.force);
    if(f(0,0)!=0. || f(0,1)!=0.) status=1;
  }
  Kokkos::finalize();
  if(status) {std::cerr<<"FAIL Kokkos grain integration\n";return 1;}
  std::cout<<"PASS Kokkos grain integration\n";
}
