#include "kokkos_grain_weighted_split.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);
 int bad=0;
 {
 using Exec=Kokkos::DefaultExecutionSpace;
 using Mem=typename Exec::memory_space;
 Kokkos::View<double*,Mem> w("w",2),p("p",2),remain("remain",2),collided("collided",2),deficit("deficit",2);
 Kokkos::View<double*[2],Mem> pre("pre",2),post("post",2),impulse("impulse",2);
 auto hw=Kokkos::create_mirror_view(w),hp=Kokkos::create_mirror_view(p);
 auto hpre=Kokkos::create_mirror_view(pre),hpost=Kokkos::create_mirror_view(post);
 for(int i=0;i<2;++i) {hw(i)=1000.;hp(i)=i==0?0.2:0.;
  hpre(i,0)=10.;hpre(i,1)=3.;hpost(i,0)=4.;hpost(i,1)=-1.;}
 Kokkos::deep_copy(w,hw);Kokkos::deep_copy(p,hp);
 Kokkos::deep_copy(pre,hpre);Kokkos::deep_copy(post,hpost);
 SPARTA_NS::GrainWeightedSplitKokkos::split_momentum<Exec>(
   w,p,pre,post,remain,collided,impulse,deficit,2.);
 auto hr=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),remain);
 auto hc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),collided);
 auto hi=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),impulse);
 auto hd=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),deficit);
 auto eq=[](double a,double b){return std::fabs(a-b)<1.e-9;};
 if(!eq(hr(0),800.)||!eq(hc(0),200.)||!eq(hi(0,0),2400.)||
    !eq(hi(0,1),1600.)||!eq(hd(0),0.5*2.*1000.*0.2*0.8*(36.+16.)))
   bad=1;
 if(!eq(hr(1),1000.)||!eq(hc(1),0.)||!eq(hi(1,0),0.)||
    !eq(hi(1,1),0.)||!eq(hd(1),0.))bad=1;
 const double before=1000.*2.*10.;
 const double after=2.*(hr(0)*10.+hc(0)*4.);
 if(!eq(before-after,hi(0,0)))bad=1;
 }
 Kokkos::finalize();
 if(bad) std::fprintf(stderr,"Kokkos weighted split bookkeeping failed\n");
 return bad;
}
