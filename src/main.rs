mod DistrWrapper;
use rand::prelude::*;
use rand_distr::Distribution;
use statrs::distribution::{Normal, Beta, Uniform, Continuous};
use std::{
    fs::File,
    io::{BufReader, BufRead},
    str::FromStr,
};

fn main() {
    let mut rng = rand::thread_rng();
    let n = 1000;
    let norm_dist_sampler = Normal::new(2., 4.).unwrap();
    let data = norm_dist_sampler.sample_iter(&mut rng).take(n).collect::<Vec<f64>>();

    // let dat_file = File::open("test_beta.csv").unwrap();
    // let buf = BufReader::new(dat_file);
    // let mut data = buf.lines()
    //     .map(|l| f64::from_str(&l.unwrap()).unwrap())
    //     .collect::<Vec<f64>>();

    // data = scale(data, 0., 1.);

    let mu_start = 1.;
    let sigma_start = 6.;
    let mu_prior = Uniform::new(f64::EPSILON, 3.).unwrap();
    let sigma_prior = Normal::new(5., 5.).unwrap();
    let mu_distr = DistrWrapper::DNormal;
    let sigma_distr = DistrWrapper::DNormal;
    let (mu_samples, sigma_samples) = sampler(&data, 10000, rng, mu_distr, sigma_distr, mu_start, sigma_start, mu_prior, sigma_prior);

    println!("d={:?}", data);
    //data.iter().for_each(|x| println!("{}", x));
    // mu_samples.iter().for_each(|x| println!("{}", x));
    // sigma_samples.iter().for_each(|x| println!("{}", x));
    println!("ms={:?}", mu_samples);
    println!("ss={:?}", sigma_samples);
}

fn scale(data: Vec<f64>, low: f64, high: f64) -> Vec<f64> {
    data.iter().map(|x| {
        (high - low) * (x - data.iter().cloned().fold(0./0., f64::min)) / (data.iter().cloned().fold(0./0., f64::max) - data.iter().cloned().fold(0./0., f64::min)) + low
    }).collect::<Vec<_>>()
}

fn sampler<T, U, V, W>(
    data: &Vec<f64>, 
    n_iter: usize, 
    mut rng: impl Rng, 
    distr_mu: T,
    distr_sigma: U,
    mut mu_current: f64, 
    mut sigma_current: f64, 
    mu_prior: V, 
    sigma_prior: W) 
-> (Vec<f64>, Vec<f64>) 
where 
    T: DistrWrapper::DWrapper + Copy,
    U: DistrWrapper::DWrapper + Copy,
    V: Continuous<f64, f64>,
    W: Continuous<f64, f64>
{
    let mut mu_samples: Vec<f64> = vec![0.; n_iter];
    let mut sigma_samples: Vec<f64> = vec![0.; n_iter];
    mu_samples[0] = mu_current;
    sigma_samples[0] = sigma_current;

    let proposal_dist = Normal::new(0.0, 0.05).unwrap();
    
    for i in 1..n_iter {
        let mut mu_proposal = mu_current + proposal_dist.sample(&mut rng);
        while mu_proposal <= 0. {
            mu_proposal = mu_current + proposal_dist.sample(&mut rng);
        }

        let distr_current = distr_mu.new(mu_current, sigma_current);
        let distr_proposal = distr_mu.new(mu_proposal, sigma_current);

        let accept = accept_reject(&distr_current, &distr_proposal, mu_current, mu_proposal, &mu_prior, data, &mut rng);

        if accept {
            mu_current = mu_proposal;
        }

        mu_samples[i] = mu_current;

        // 
        
        let mut sigma_proposal = sigma_current + proposal_dist.sample(&mut rng);
        while sigma_proposal <= 0. {
            sigma_proposal = sigma_current + proposal_dist.sample(&mut rng);
        }
        
        let distr_current = distr_sigma.new(mu_current, sigma_current);
        let distr_proposal = distr_sigma.new(mu_current, sigma_proposal);

        let accept = accept_reject(&distr_current, &distr_proposal, sigma_current, sigma_proposal, &sigma_prior, data, &mut rng);

        if accept {
            sigma_current = sigma_proposal;
        }

        sigma_samples[i] = sigma_current;
    }

    (mu_samples, sigma_samples)
}

fn accept_reject<T: Continuous<f64, f64>, U: Continuous<f64, f64>>(distr_current: &T, distr_proposal: &T, param_current: f64, param_proposal: f64, distr_prior: &U, data: &Vec<f64>, mut rng: impl Rng) -> bool {
    let likelihood_current = calc_loglikelihood(distr_current, data);
    let likelihood_proposal = calc_loglikelihood(distr_proposal, data);

    let prior_current = distr_prior.pdf(param_current).ln();
    let prior_proposal = distr_prior.pdf(param_proposal).ln();

    let p_current = likelihood_current + prior_current;
    let p_proposal = likelihood_proposal + prior_proposal;

    let p_accept = f64::min((p_proposal - p_current).exp(), 1.);
    rng.gen_bool(p_accept)
}

fn calc_loglikelihood<T: Continuous<f64, f64>>(distr: &T, data: &Vec<f64>) -> f64 {
    data.iter().map(|x| {
        distr.pdf(*x).ln()
    }).sum()
}
