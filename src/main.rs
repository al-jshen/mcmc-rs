use rand::prelude::*;
use rand_distr::Distribution;
use statrs::distribution::{Normal, Beta, Uniform, Continuous};
use std::{
    fs::File,
    io::{BufReader, BufRead},
    str::FromStr,
};

fn main() {
    let n = 1000;
    let mut rng = rand::thread_rng();
    let norm_dist_sampler = Normal::new(7., 3.).unwrap();
    let data = norm_dist_sampler.sample_iter(&mut rng).take(n).collect::<Vec<f64>>();

    // let mut dat_file = File::open("data.csv").unwrap();
    // let buf = BufReader::new(dat_file);
    // let mut data = buf.lines()
    //     .map(|l| f64::from_str(&l.unwrap()).unwrap())
    //     .collect::<Vec<f64>>();

    // data = scale(data, 0., 1.);

    let prior_dist = Uniform::new(6., 8.).unwrap();
    let mu_start = 8.;
    let samples = sampler(&data, 2000, rng, prior_dist, mu_start);

    println!("{}", n);
    data.iter().for_each(|x| println!("{}", x));
    samples.iter().for_each(|x| println!("{}", x));
    //println!("s={:?}", samples);
}

fn scale(data: Vec<f64>, low: f64, high: f64) -> Vec<f64> {
    data.iter().map(|x| {
        (high - low) * (x - data.iter().cloned().fold(0./0., f64::min)) / (data.iter().cloned().fold(0./0., f64::max) - data.iter().cloned().fold(0./0., f64::min)) + low
    }).collect::<Vec<_>>()
}

fn sampler(data: &Vec<f64>, n_iter: usize, mut rng: ThreadRng, prior_dist: Uniform, mut mu_current: f64) -> Vec<f64>{
    let mut samples: Vec<f64> = vec![0.; n_iter];
    samples[0] = mu_current;

    let proposal_dist = Normal::new(0., 0.05).unwrap();
    
    for i in 1..n_iter {
        let mu_proposal = mu_current + proposal_dist.sample(&mut rng);

        let likelihood_current: f64 = data.iter().map(|x| {
            Normal::new(mu_current, 1.).unwrap()
                .pdf(*x)
                .ln()
        }).sum();
        let likelihood_proposal: f64 = data.iter().map(|x| {
            Normal::new(mu_proposal, 1.).unwrap()
                .pdf(*x)
                .ln()
        }).sum();

        
        let prior_current = prior_dist.pdf(mu_current).ln();
        let prior_proposal = prior_dist.pdf(mu_proposal).ln();

        let p_current = likelihood_current + prior_current;
        let p_proposal = likelihood_proposal + prior_proposal;


        let p_accept = f64::min((p_proposal - p_current).exp(), 1.);
        let accept = rng.gen_bool(p_accept);

        if accept {
            mu_current = mu_proposal;
        }

        samples[i] = mu_current
    }
    samples
}
