use rand_distr::Distribution;
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

#[derive(Debug, Clone, Copy)]
pub struct Wrap<T, S>
where
    T: Continuous<f64, f64>,
    S: Distribution<f64>,
{
    distr: T,
    sampler: S,
}

impl<T, S> Wrap<T, S>
where
    T: Continuous<f64, f64>,
    S: Distribution<f64>,
{
    pub fn new(distr: T, sampler: S) -> Self {
        Wrap { distr, sampler }
    }
    pub fn pdf(self, x: f64) -> f64 {
        self.distr.pdf(x)
    }
    pub fn sample<R>(self, rng: &mut R) -> f64
    where
        R: rand::Rng, // + ?Sized,
    {
        self.sampler.sample(rng)
    }
}

// eventually extend this to continuous AND discrete distributions
//
// enum Foo<T: ATrait, U: OtherTrait> {
//     Foo(T),
//     Bar(U)
// }

// fn foo<T: ATrait, U: OtherTrait>(param: Foo<T, U>) -> () {
// }
