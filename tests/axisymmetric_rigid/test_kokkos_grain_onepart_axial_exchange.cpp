#include "kokkos_grain_onepart_axial_exchange.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
struct MockPart {int ispecies; double weight; double v[3];};
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);
 int fail=0;
 {
 using Ex=Kokkos::DefaultExecutionSpace;
 using Mem=typename Ex::memory_space;
 Kokkos::View<MockPart*,Mem> particles("parts",2);
 Kokkos::View<double*,Mem> species("species",1),prob("prob",2),after("after",2);
 Kokkos::View<int*,Mem> target("target",2);
 Kokkos::View<double*[2],Mem> impulse("impulse",1);
 auto h=Kokkos::create_mirror_view(particles);
 auto s=Kokkos::create_mirror_view(species);
 auto p=Kokkos::create_mirror_view(prob);
 auto v=Kokkos::create_mirror_view(after);
 auto t=Kokkos::create_mirror_view(target);
 s(0)=2.;
 for(int i=0;i<2;i++) {
   h(i).ispecies=0; h(i).weight=3.; h(i).v[0]=10.;
   h(i).v[1]=4.; h(i).v[2]=0.; t(i)=0;
 }
 p(0)=0.25; p(1)=0.; v(0)=6.; v(1)=-100.;
 Kokkos::deep_copy(particles,h);Kokkos::deep_copy(species,s);
 Kokkos::deep_copy(prob,p);Kokkos::deep_copy(after,v);
 Kokkos::deep_copy(target,t);Kokkos::deep_copy(impulse,0.);
 // A corrupt probability must be rejected before modifying gas state.
 p(0)=1.25;Kokkos::deep_copy(prob,p);
 bool rejected=false;
 try {
   SPARTA_NS::AxisymmetricGrains::checked_onepart_axial_partial_exchange<Ex>(
     particles,species,prob,target,after,impulse,2,5.,true);
 } catch(const std::invalid_argument&) { rejected=true; }
 if(!rejected) fail=1;
 p(0)=0.25;Kokkos::deep_copy(prob,p);
 SPARTA_NS::AxisymmetricGrains::checked_onepart_axial_partial_exchange<Ex>(
   particles,species,prob,target,after,impulse,2,5.,true);
 Kokkos::deep_copy(h,particles);
 auto j=Kokkos::create_mirror_view(impulse);
 Kokkos::deep_copy(j,impulse);
 // 5 * species_mass(2) * cellweight(3) = 30; delta_v=-1
 if(std::abs(h(0).v[0]-9.)>1.e-12 ||
    std::abs(h(0).v[1]-4.)>1.e-12 ||
    std::abs(h(1).v[0]-10.)>1.e-12 ||
    std::abs(j(0,0)-30.)>1.e-12) fail=1;
 }
 Kokkos::finalize();
 if(fail) {std::cerr<<"FAIL onepart axial partial exchange\n";return 1;}
 std::cout<<"PASS SPARTA-style OnePart axial partial exchange\n";
}
