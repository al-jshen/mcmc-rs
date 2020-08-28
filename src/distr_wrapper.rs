use statrs::distribution::*;

#[derive(Debug, Clone, Copy)]
pub struct DBeta;
#[derive(Debug, Clone, Copy)]
pub struct DNormal;
#[derive(Debug, Clone, Copy)]
pub struct DUniform;
#[derive(Debug, Clone, Copy)]
pub struct DGamma;

pub trait DWrapper {
    type Distr: Continuous<f64, f64>;
    fn new(self, param1: f64, param2: f64) -> Self::Distr;
}

impl DWrapper for DBeta {
    type Distr = Beta;
    fn new(self, param1: f64, param2: f64) -> Self::Distr {
        Beta::new(param1, param2).unwrap()
    }
}

impl DWrapper for DNormal {
    type Distr = Normal;
    fn new(self, param1: f64, param2: f64) -> Self::Distr {
        Normal::new(param1, param2).unwrap()
    }
}

impl DWrapper for DUniform {
    type Distr = Uniform;
    fn new(self, param1: f64, param2: f64) -> Self::Distr {
        Uniform::new(param1, param2).unwrap()
    }
}

impl DWrapper for DGamma {
    type Distr = Gamma;
    fn new(self, param1: f64, param2: f64) -> Self::Distr {
        Gamma::new(param1, param2).unwrap()
    }
}
