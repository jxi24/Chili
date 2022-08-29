#pragma once

#include "Tools/FourVector.hh"

namespace apes {

  double SqLam(double,double,double);
  FourVector LT(const FourVector &,const FourVector &,const FourVector &);

  // double PeakedDist(double,double,double,double,int,double);
  // double PeakedWeight(double,double,double,double,double,int,double&);

  double MasslessPropMomenta(double,double,double);
  double MasslessPropWeight(double,double,double,double&);

  double MassivePropMomenta(double,double,double,double,double);
  double MassivePropWeight(double,double,double,double,double,double&);

  // double ThresholdMomenta(double,double,double,double,double);
  // double ThresholdWeight(double,double,double,double,double,double&);

  void SChannelMomenta(FourVector,double,double,
		       FourVector&,FourVector&,
		       double,double,double=-1.0,double=1.0,
		       const FourVector &x=FourVector(1.,1.,0.,0.));
  double SChannelWeight(const FourVector&,const FourVector&,
			double&,double&,double=-1.0,double=1.0,
			const FourVector &x=FourVector(1.,1.,0.,0.));
}// end of namespace apes
