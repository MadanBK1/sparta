/* Prototype-only device bookkeeping for fractional collision participation.
   Two separately weighted populations retain the postcollision kinetic
   variance lost by an averaged-velocity update. No SPARTA particle is
   created here. A physical scattering law and safe SPARTA weight API are
   still required before activating two-way coupling.
*/
#ifndef SPARTA_KOKKOS_GRAIN_WEIGHTED_SPLIT_H
#define SPARTA_KOKKOS_GRAIN_WEIGHTED_SPLIT_H
#include <Kokkos_Core.hpp>
namespace SPARTA_NS { namespace GrainWeightedSplitKokkos {
template<class ExecSpace,class WeightView,class FractionView,class PreView,
         class PostView,class RemainView,class CollideView,class ImpulseView,
         class DeficitView>
void split_momentum(const WeightView& weight,const FractionView& fraction,
                    const PreView& pre,const PostView& post,
                    const RemainView& remainder,const CollideView& collided,
                    const ImpulseView& grain_impulse,const DeficitView& deficit,
                    double molecule_mass) {
  const int n=static_cast<int>(weight.extent(0));
  Kokkos::parallel_for("grain_weighted_split_bookkeeping",
    Kokkos::RangePolicy<ExecSpace>(0,n),KOKKOS_LAMBDA(const int i) {
      const double w=weight(i),p=fraction(i);
      if(!(w>0. && p>=0. && p<=1. && molecule_mass>0.)) {
        remainder(i)=-1.;collided(i)=-1.;
        grain_impulse(i,0)=0.;grain_impulse(i,1)=0.;deficit(i)=-1.;
        return;
      }
      remainder(i)=w*(1.-p);
      collided(i)=w*p;
      double delta2=0.;
      for(int d=0;d<2;++d) {
        const double dv=post(i,d)-pre(i,d);
        grain_impulse(i,d)=-molecule_mass*collided(i)*dv;
        delta2+=dv*dv;
      }
      deficit(i)=0.5*molecule_mass*w*p*(1.-p)*delta2;
    });
}
}} // namespaces
#endif
