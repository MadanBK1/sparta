#include "kokkos_grain_annular_encounter.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
int main(int argc,char** argv) {
  Kokkos::initialize(argc,argv);
  int failed=0;
  {
    typedef Kokkos::DefaultExecutionSpace Exec;
    typedef typename Exec::memory_space Mem;
    Kokkos::View<double*,Mem> speed("speed",3), weight("weight",3), result("expected",3);
    auto hs=Kokkos::create_mirror_view(speed);
    auto hw=Kokkos::create_mirror_view(weight);
    hs(0)=0.; hs(1)=10.; hs(2)=10.;
    hw(0)=1.e6; hw(1)=1.e6; hw(2)=2.e6;
    Kokkos::deep_copy(speed,hs);
    Kokkos::deep_copy(weight,hw);
    const double r=75.e-6, vol=1.e-5, dt=1.e-6;
    SPARTA_NS::GrainAnnularKokkos::expected_counts<Exec>(speed,weight,result,r,vol,dt);
    auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
    const double tau=std::acos(-1.)*r*r*10.*dt/vol;
    const double expected=1.e6*(-std::expm1(-tau));
    if(h(0)!=0. || std::fabs(h(1)-expected)>expected*1.e-9 ||
       std::fabs(h(2)-2.*expected)>2.*expected*1.e-9) failed=1;
    // Expected physical impacts can be > 1 and are NOT the probability
    // that one weighted DSMC representative experiences a full collision.
    if(!(h(1)>1.0)) failed=1;
  }
  Kokkos::finalize();
  if(failed) std::fprintf(stderr,"Kokkos annular encounter expectation test failed\\n");
  return failed;
}
