#[macro_use]
mod distr_wrapper;
mod sampler;
mod utils;
// use rand::prelude::*;
// use rand_distr::Distribution;
use rayon::prelude::*;
// use statrs::distribution::*;
use statistics::distributions::*;
use std::time::Instant;
// use std::{
//     fs::File,
//     io::{BufRead, BufReader},
//     str::FromStr,
// };

fn main() {
    // let mut rng = rand::thread_rng();
    let n = 5000;
    let data = Normal::new(5., 6.).sample_iter(n);

    // let test_wrapper = distr_wrapper::Wrap::new(
    //     Normal::new(2., 3.).unwrap(),
    //     rand_distr::Normal::new(2., 3.).unwrap(),
    // );

    // println!("{:?}", test_wrapper);
    // println!("{}", test_wrapper.sample(&mut rng));
    // println!(
    //     "{:?}",
    //     test_wrapper.sample_iter(rng).take(5).collect::<Vec<_>>()
    // );
    // println!("{}", test_wrapper.pdf(2.));

    let proposal_dist = Normal::default();

    let priors: Vec<Box<dyn Continuous>> = vec![
        Box::new(Normal::new(4., 2.)),
        Box::new(Uniform::new(0., 10.)),
    ];

    let proposals: Vec<Box<dyn Continuous>> = vec![
        Box::new(Normal::new(0., 0.1)),
        Box::new(Normal::new(0., 0.1)),
    ];

    let n_iter = 3000;

    let (samples, acceptances) = sampler::sampler(&data, n_iter, proposal_dist, priors, proposals);

    println!("{:?}", samples);
    println!("{:?}", acceptances);
    // let chains = (0..num_cpus::get())
    //     .into_par_iter()
    //     .map(|i| {
    //         let now = Instant::now();
    //         let (samples, acceptance_rates) =
    //             sampler::sampler(&data, n_iter as usize, proposal_dist, priors, proposals);
    //         eprintln!(
    //             "chain: {} \t time: {:.3}s \t mu accept rate: {:.3} \t sigma accept rate: {:.3}",
    //             i,
    //             now.elapsed().as_secs_f64(),
    //             mu_accept as f64 / n_iter as f64,
    //             sigma_accept as f64 / n_iter as f64
    //         );
    //         (mu, sigma)
    //     })
    //     .collect::<Vec<(Vec<f64>, Vec<f64>)>>();

    // println!("d={:?}", data);
    // println!("chains={:?}", chains);
}
