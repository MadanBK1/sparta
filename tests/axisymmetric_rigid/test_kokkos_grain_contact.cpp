#include "kokkos_axisymmetric_grains.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);
  int status=0;
  {
    using Exec=Kokkos::DefaultExecutionSpace;
    using Grains=SPARTA_NS::AxisymmetricGrains::State<Exec>;
    Grains g(2);
    auto x=Kokkos::create_mirror_view(g.position);
    auto v=Kokkos::create_mirror_view(g.velocity);
    auto rad=Kokkos::create_mirror_view(g.radius);
    x(0,0)=0.;x(0,1)=0.;x(1,0)=0.9;x(1,1)=0.;
    for(int i=0;i<2;i++) {
      v(i,0)=0.;v(i,1)=0.;rad(i)=0.5;
    }
    Kokkos::deep_copy(g.position,x);
    Kokkos::deep_copy(g.velocity,v);
    Kokkos::deep_copy(g.radius,rad);
    Kokkos::View<int*[2],typename Exec::memory_space> pairs("pairs",1);
    auto h=Kokkos::create_mirror_view(pairs);
    h(0,0)=0;h(0,1)=1;Kokkos::deep_copy(pairs,h);
    SPARTA_NS::AxisymmetricGrains::zero_forces(g);
    SPARTA_NS::AxisymmetricGrains::linear_normal_contacts(g,pairs,100.,0.);
    auto f=Kokkos::create_mirror_view(g.force);
    Kokkos::deep_copy(f,g.force);
    if (std::abs(f(0,0)+10.)>1e-10 ||
        std::abs(f(1,0)-10.)>1e-10 ||
        std::abs(f(0,0)+f(1,0))>1e-12 ||
        std::abs(f(0,1))>1e-12 || std::abs(f(1,1))>1e-12)
      status=1;
  }
  Kokkos::finalize();
  if(status) {std::cerr<<"FAIL Kokkos normal contact\n";return 1;}
  std::cout<<"PASS Kokkos normal contact and equal/opposite forces\n";
}
