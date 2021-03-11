mod sampler;
mod sampler_linreg;
mod utils;

use compute::distributions::*;
use compute::summary::mean;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const ALPHA: f64 = 5.;
const BETA: f64 = 2.;
const SIGMA: f64 = 10.;

fn main() {
    // let n = 50000;
    // let xdata = (0..500).map(f64::from).collect::<Vec<_>>();
    // let alpha = 5.;
    // let beta = 2.;
    // let sigma = Normal::new(0., 20.);
    // let data = &xdata
    //     .iter()
    //     .map(|x| alpha + beta * x + sigma.sample())
    //     .collect::<Vec<_>>();

    let dist_fn = Normal::default();
    // let scatter = Normal::new(0., SIGMA);
    // let xs = (0..=3).map(|x| x as f64).collect::<Vec<_>>();
    // let ys = &xs
    //     .iter()
    //     .map(|x| ALPHA + BETA * x + scatter.sample())
    //     .collect::<Vec<_>>();

    let n = 5000;
    let data = Normal::new(5., 6.).sample_iter(n);

    let priors: Arc<Vec<Box<dyn Continuous>>> = Arc::new(vec![
        Box::new(Normal::new(4., 5.)),
        Box::new(Normal::new(3., 3.)),
        // Box::new(Exponential::new(1. / 15.)),
    ]);

    let proposals: Arc<Vec<Box<dyn Continuous>>> = Arc::new(vec![
        Box::new(Normal::new(0., 0.5)),
        Box::new(Normal::new(0., 0.5)),
        // Box::new(Normal::new(0., 0.5)),
    ]);

    let n_iter = 5000;

    // let samples = sampler_linreg::sampler(
    //     &xs,
    //     &ys,
    //     n_iter,
    //     Arc::clone(&priors),
    //     Arc::clone(&proposals),
    // );
    let chains = (0..num_cpus::get())
        .into_par_iter()
        .map(|i| {
            let now = Instant::now();
            let (samples, acceptance_rates) = sampler::sampler(
                &data,
                n_iter,
                dist_fn,
                Arc::clone(&priors),
                Arc::clone(&proposals),
            );
            eprint!(
                "chain: {} \t time: {:.3}s \t",
                i,
                now.elapsed().as_secs_f64()
            );
            for (idx, rate) in acceptance_rates.iter().enumerate() {
                eprint!("p{}: {:.1}%\t", idx, *rate as f64 / n_iter as f64 * 100.);
            }
            eprintln!("");
            samples
        })
        .collect::<Vec<Vec<Vec<f64>>>>();

    println!("d={:?}", data);
    println!("chains={:?}", chains);

    // println!("{:?}", samples);
    // for p in samples {
    //     println!("{}", mean(&p[(n_iter * 4 / 5)..]));
    // }
}
