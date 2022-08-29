#include "Tools/ChannelElements.hh"
#include "Tools/Poincare.hh"
#include "Tools/ThreeVector.hh"
#include "Tools/Utilities.hh"
#include "spdlog/spdlog.h"

using namespace apes;

FourVector apes::LT(const FourVector &a,const FourVector &b,const FourVector &c) {
  double t(a[1]*b[2]*c[3]+a[2]*b[3]*c[1]+a[3]*b[1]*c[2]
	   -a[1]*b[3]*c[2]-a[3]*b[2]*c[1]-a[2]*b[1]*c[3]);
  double x(-a[0]*b[2]*c[3]-a[2]*b[3]*c[0]-a[3]*b[0]*c[2]
	   +a[0]*b[3]*c[2]+a[3]*b[2]*c[0]+a[2]*b[0]*c[3]);
  double y(-a[1]*b[0]*c[3]-a[0]*b[3]*c[1]-a[3]*b[1]*c[0]
	   +a[1]*b[3]*c[0]+a[3]*b[0]*c[1]+a[0]*b[1]*c[3]);
  double z(-a[1]*b[2]*c[0]-a[2]*b[0]*c[1]-a[0]*b[1]*c[2]
	   +a[1]*b[0]*c[2]+a[0]*b[2]*c[1]+a[2]*b[1]*c[0]);
  return FourVector(t,-x,-y,-z);
}

double apes::SqLam(double s,double s1,double s2) {
  double arg(sqr(s-s1-s2)-4.*s1*s2);
  if (arg>0.) return sqrt(arg)/s;
  return 0.;
}

double apes::MasslessPropWeight(double smin,double smax,const double s,double &ran) {
  if (s<smin || s>smax)
   spdlog::error("MasslessPropWeight(): Value out of bounds: {} .. {} vs. {}",smin,smax,s);
  double hmin = 1/smax;
  double hmax = 1/smin;
  double delh = hmax-hmin;
  ran = (1.0/s-hmin)/delh;
  double w = delh*pow(s, 3);
  if (IsBad(w)) spdlog::error("MasslessPropWeight(): Weight is {}",w);
  return 1./w;
}

double apes::MasslessPropMomenta(double smin,double smax, double ran) {
  double hmin = 1/smax;
  double hmax = 1/smin;
  double delh = hmax-hmin;
  double s = 1.0/(delh*ran+hmin);
  if (IsBad(s)) spdlog::error("MasslessPropMomenta(): Value is {}",s);
  return s;
}

double apes::MassivePropWeight(double m,double g,double smin,double smax,double s,double &ran) {
  if (s<smin || s>smax)
    spdlog::error("MassivePropWeight(): Value out of bounds: {} .. {} vs. {}",smin,smax,s);
  double m2(m*m), mw(m*g);
  double ymax(atan((smin-m2)/mw)), ymin(atan((smax-m2)/mw));
  double y(atan((s-m2)/mw));
  ran=(y-ymin)/(ymax-ymin);
  double w(mw/((s-m2)*(s-m2)+mw*mw));
  w=(ymin-ymax)/w;
  if (IsBad(w)) spdlog::error("MassivePropWeight(): Weight is {}",w);
  return 1./w;
}

double apes::MassivePropMomenta
(double m,double g,double smin,double smax,double ran)
{
  double m2(m*m), mw(m*g), s;
  double ymax(atan((smin-m2)/mw)), ymin(atan((smax-m2)/mw));
  s=m2+mw*tan(ymin+ran*(ymax-ymin));
  if (IsBad(s)) spdlog::error("MassivePropMomenta(): Value is {}",s);
  return s;
}

// double apes::ThresholdWeight
// (double sexp,double m,double smin,double smax,double s,double &ran)
// {
//   if (s<smin || s>smax) spdlog::error("ThresholdWeight(): Value out of bounds: {} .. {} vs. {}",smin,smax,s);
//   double m2(m*m), sg(sqrt(s*s+m2*m2));
//   double sgmin(sqrt(smin*smin+m2*m2)), sgmax(sqrt(smax*smax+m2*m2));
//   double w=PeakedWeight(0.,sexp,sgmin,sgmax,sg,1,ran)/(s*pow(sg,-sexp-1.));
//   if (IsBad(w)) spdlog::error("ThresholdWeight(): Weight is {}",w);
//   return 1./w;
// }
// 
// double apes::ThresholdMomenta
// (double sexp,double m,double smin,double smax,double ran)
// {
//   double m2(m*m);
//   double sgmin(sqrt(smin*smin+m2*m2)), sgmax(sqrt(smax*smax+m2*m2));
//   double s(sqrt(sqr(PeakedDist(0.,sexp,sgmin,sgmax,1,ran))-m2*m2));
//   if (IsBad(s)) spdlog::error("ThresholdMomenta(): Value is {}",s);
//   return s;
// }

void apes::SChannelMomenta
(FourVector p,double s1,double s2,FourVector &p1,FourVector &p2,double ran1,
 double ran2,double ctmin,double ctmax,const FourVector &_xref)
{
  double s(p.Mass2()), rs(sqrt(std::abs(s)));
  double e1((s+s1-s2)/rs/2.), m1(sqrt(e1*e1-s1));
  double ct(ctmin+(ctmax-ctmin)*ran1), st(sqrt(1.-ct*ct));
  double phi(2.*M_PI*ran2);
  FourVector xref(_xref[0]<0.0?-_xref:_xref);
  FourVector pl(p.P2()?p:FourVector(1.,0.,0.,1.));
  Poincare cms(p), zax(pl,xref);
  FourVector n_perp(zax.PT()), l_perp(LT(pl,xref,n_perp));
  l_perp*=1.0/sqrt(std::abs(l_perp.Mass2()));
  p1=FourVector(m1*ct*pl.Vec3()/pl.P(),e1);
  p1+=m1*st*(cos(phi)*l_perp+sin(phi)*n_perp);
  cms.BoostBack(p1);
  p2=p-p1;
}

double apes::SChannelWeight
(const FourVector &p1, const FourVector &p2,double &ran1, double &ran2,
 double ctmin, double ctmax,const FourVector &_xref)
{
  FourVector p(p1+p2), p1h(p1);
  FourVector xref(_xref[0]<0.?-_xref:_xref);
  FourVector pl(p.P2()?p:FourVector(1.,0.,0.,1.));
  Poincare cms(p), zax(pl,xref);
  FourVector n_perp(zax.PT()), l_perp(LT(pl,xref,n_perp));
  l_perp*=1.0/sqrt(std::abs(l_perp.Mass2()));
  cms.Boost(p1h);
  double ct(p1h.Vec3()*pl.Vec3()/sqrt(p1h.P2()*pl.P2()));
  double cp(-l_perp*p1), sp(-n_perp*p1), norm(sqrt(cp*cp+sp*sp));
  cp/=norm;
  sp/=norm;
  ran1=(ct-ctmin)/(ctmax-ctmin);
  ran2=atan2(sp,cp)/(2.*M_PI);
  if (ran2<0.) ran2+=1.;
  double w((ctmax-ctmin)/2.);
  w*=M_PI*SqLam(p.Mass2(),p1.Mass2(),p2.Mass2())/2.;
  if (IsBad(w)) spdlog::error("SChannelWeight(): Weight is {}.",w);
  return 1./w;
}
