use statrs::distribution::*;

#[derive(Debug, Clone, Copy)]
pub struct DBeta;
#[derive(Debug, Clone, Copy)]
pub struct DNormal;
#[derive(Debug, Clone, Copy)]
pub struct DUniform;
#[derive(Debug, Clone, Copy)]
pub struct DGamma;
#[derive(Debug, Clone, Copy)]
pub struct DLogNormal;
#[derive(Debug, Clone, Copy)]
pub struct DExponential;
#[derive(Debug, Clone, Copy)]
pub struct DChiSquared;
#[derive(Debug, Clone, Copy)]
pub struct DStudentsT;

pub trait DWrapper {
    type Distr: Continuous<f64, f64>;
    fn new(self, param1: f64, param2: f64) -> Self::Distr;
}

macro_rules! impl_wrap {
    ($dt: ty, $t: ty) => {
        impl DWrapper for $dt {
            type Distr = $t;
            fn new(self, param1: f64, param2: f64) -> Self::Distr {
                Self::Distr::new(param1, param2).unwrap()
            }
        }
    };
}

impl_wrap!(DBeta, Beta);
impl_wrap!(DNormal, Normal);
impl_wrap!(DUniform, Uniform);
impl_wrap!(DGamma, Gamma);
impl_wrap!(DLogNormal, LogNormal);
