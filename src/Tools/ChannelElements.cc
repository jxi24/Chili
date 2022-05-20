#include "Tools/Channel_Elements.hh"
#include "Tools/Poincare.hh"
#include "Tools/ThreeVector.hh"
#include "Tools/Utilities.hh"
#include "spdlog/spdlog.h"

using namespace apes;

FourVector apes::LT(const FourVector &a,const FourVector &b,const FourVector &c)
{
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

double apes::SqLam(double s,double s1,double s2)
{
  double arg(sqr(s-s1-s2)-4.*s1*s2);
  if (arg>0.) return sqrt(arg)/s;
  return 0.;
}

double apes::PeakedDist(double a,double cn,double cxm,double cxp,int k,double ran)
{
  double ce(1.-cn);
  if (ce!=0.) return k*(pow(ran*pow(a+k*cxp,ce)+(1.-ran)*pow(a+k*cxm,ce),1/ce)-a);
  return k*((a+k*cxm)*pow((a+k*cxp)/(a+k*cxm),ran)-a);
}

double apes::PeakedWeight(double a,double cn,double cxm,double cxp,double res,int k,double &ran)
{
  double ce(1.-cn), w;
  if (ce!=0.) {
    double amin=pow(a+k*cxm,ce);
    w=pow(a+k*cxp,ce)-amin;
    ran=(pow(a+k*res,ce)-amin)/w;
    w/=k*ce;
  }
  else {
    double amin=a+k*cxm;
    w=log((a+k*cxp)/amin);
    ran=log((a+k*res)/amin)/w;
    w/=k;
  }
  return w;
}

double apes::MasslessPropWeight
(double sexp,double smin,double smax,const double s,double &ran)
{
  if (s<smin || s>smax) spdlog::error("MasslessPropWeight(): Value out of bounds: {} .. {} vs. {}",smin,smax,s);
  double w(PeakedWeight(0.,sexp,smin,smax,s,1,ran)/pow(s,-sexp));
  if (IsBad(w)) spdlog::error("MasslessPropWeight(): Weight is {}",w);
  return 1./w;
}

double apes::MasslessPropMomenta
(double sexp,double smin,double smax, double ran)
{
  double s(PeakedDist(0.,sexp,smin,smax,1,ran));
  if (IsBad(s)) spdlog::error("MasslessPropMomenta(): Value is {}",s);
  return s;
}

double apes::MassivePropWeight
(double m,double g,double smin,double smax,double s,double &ran)
{
  if (s<smin || s>smax) spdlog::error("MassivePropWeight(): Value out of bounds: {} .. {} vs. {}",smin,smax,s);
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

double apes::ThresholdWeight
(double sexp,double m,double smin,double smax,double s,double &ran)
{
  if (s<smin || s>smax) spdlog::error("ThresholdWeight(): Value out of bounds: {} .. {} vs. {}",smin,smax,s);
  double m2(m*m), sg(sqrt(s*s+m2*m2));
  double sgmin(sqrt(smin*smin+m2*m2)), sgmax(sqrt(smax*smax+m2*m2));
  double w=PeakedWeight(0.,sexp,sgmin,sgmax,sg,1,ran)/(s*pow(sg,-sexp-1.));
  if (IsBad(w)) spdlog::error("ThresholdWeight(): Weight is {}",w);
  return 1./w;
}

double apes::ThresholdMomenta
(double sexp,double m,double smin,double smax,double ran)
{
  double m2(m*m);
  double sgmin(sqrt(smin*smin+m2*m2)), sgmax(sqrt(smax*smax+m2*m2));
  double s(sqrt(sqr(PeakedDist(0.,sexp,sgmin,sgmax,1,ran))-m2*m2));
  if (IsBad(s)) spdlog::error("ThresholdMomenta(): Value is {}",s);
  return s;
}

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

void apes::TChannelMomenta
(FourVector p1in,FourVector p2in,FourVector &p1out,FourVector &p2out,
 double s1out,double s2out,double mt,double ctexp,
 double ctmax,double ctmin,double ran1,double ran2)
{
  FourVector pin(p1in+p2in);
  double s(pin.Mass2()), rs(sqrt(std::abs(s)));
  double s1in(p1in.Mass2()), s2in(p2in.Mass2());
  double e1in((s+s1in-s2in)/2./rs), m1in(sqrt(e1in*e1in-s1in));
  double e1out((s+s1out-s2out)/2./rs), m1out(sqrt(e1out*e1out-s1out));
  double a=(mt*mt-s1in-s1out+2.*e1in*e1out)/(2.*m1in*m1out);
  if (a<=1.0+1.0e-6) a=1.0+1.0e-6;
  double aminct(PeakedDist(0.,ctexp,a-ctmax,a-ctmin,1,ran1));
  double ct(a-aminct), st(sqrt(1.-ct*ct));
  double phi(2.*M_PI*ran2);
  p1out=FourVector(m1out*ThreeVector(st*cos(phi),st*sin(phi),ct),e1out); 
  Poincare cms(pin);
  cms.Boost(p1in);
  Poincare zax(p1in,FourVector(1.,0.,0.,1.));
  zax.RotateBack(p1out);
  cms.BoostBack(p1out);
  p2out=pin-p1out;
}

double apes::TChannelWeight
(const FourVector &p1in,const FourVector &p2in,const FourVector &p1out,const FourVector &p2out,
 double mt,double ctexp,double ctmax,double ctmin,double &ran1,double &ran2)
{
  FourVector pin(p1in+p2in), p1inh(p1in), p1outh(p1out);
  double s(pin.Mass2()), rs(sqrt(std::abs(s)));
  double s1in(p1in.Mass2()), s2in(p2in.Mass2());
  double s1out(p1out.Mass2()), s2out(p2out.Mass2());
  double e1in((s+s1in-s2in)/2./rs), m1in(sqrt(e1in*e1in-s1in));
  double e1out((s+s1out-s2out)/2./rs), m1out(sqrt(e1out*e1out-s1out));
  double a=(mt*mt-s1in-s1out+2.*e1in*e1out)/(2.*m1in*m1out);
  if (a<=1.0+1.0e-6) a=1.0+1.0e-6;
  Poincare cms(pin);
  cms.Boost(p1inh);
  Poincare zax(p1inh,FourVector(1.,0.,0.,1.));
  cms.Boost(p1outh);
  zax.Rotate(p1outh);
  double pa1(pow(a-ctmax,1.-ctexp));
  double ct(p1outh[3]/p1outh.P());
  if (ct<ctmin || ct>ctmax) {
    spdlog::error("TChannelWeight(): Error in momentum mapping.");
    ran1=ran2=-1.;
    return 0.;
  }
  ran1=(pow(a-ct,1.-ctexp)-pa1);
  ran1/=(pow(a-ctmin,1.-ctexp)-pa1);
  ran2=asin(p1outh[2]/p1outh.Pt())/(2.*M_PI);
  if (p1outh[1]<0.) ran2=.5-ran2;
  if (ran2<0.) ran2+=1.;
  double aminct(a-ct);
  double w(PeakedWeight(0.,ctexp,a-ctmax,a-ctmin,aminct,1,ran1));
  w*=m1out*M_PI/(2.*rs)/pow(aminct,-ctexp);
  if (IsBad(w)) spdlog::error("TChannelWeight(): Weight is {}.",w);
  return 1./w;
}
