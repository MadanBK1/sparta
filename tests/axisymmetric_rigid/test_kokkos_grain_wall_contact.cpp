#include "kokkos_axisymmetric_grains.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
int main(int argc,char**argv) {
  Kokkos::initialize(argc,argv);
  int fail=0;
  {
    using Exec=Kokkos::DefaultExecutionSpace;
    SPARTA_NS::AxisymmetricGrains::State<Exec> g(2);
    auto p=Kokkos::create_mirror_view(g.position);
    auto v=Kokkos::create_mirror_view(g.velocity);
    auto r=Kokkos::create_mirror_view(g.radius);
    // A wall at x=0; allowed side x>0; one penetrating grain.
    p(0,0)=0.3; p(0,1)=0.2;
    p(1,0)=1.0; p(1,1)=0.2;
    v(0,0)=-1.0;v(0,1)=0.0;
    v(1,0)=0.0; v(1,1)=0.0;
    r(0)=r(1)=0.5;
    Kokkos::deep_copy(g.position,p);Kokkos::deep_copy(g.velocity,v);
    Kokkos::deep_copy(g.radius,r);
    SPARTA_NS::AxisymmetricGrains::zero_forces(g);
    SPARTA_NS::AxisymmetricGrains::linear_plane_contacts(g,1.,0.,0.,100.,3.);
    auto f=Kokkos::create_mirror_view(g.force);
    Kokkos::deep_copy(f,g.force);
    if (std::abs(f(0,0)-23.)>1e-12 ||
        std::abs(f(0,1))>1e-12 ||
        std::abs(f(1,0))>1e-12 ||
        std::abs(f(1,1))>1e-12) fail=1;
  }
  Kokkos::finalize();
  if(fail) {std::cerr<<"FAIL normal plane contact\n";return 1;}
  std::cout<<"PASS normal plane contact\n";
}
