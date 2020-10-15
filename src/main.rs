mod sampler;
mod utils;
use compute::distributions::*;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

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

    // let dist_fn = Normal::default();

    let n = 5000;
    let data = Normal::new(5., 6.).sample_iter(n);

    let proposal_dist = Normal::default();

    let priors: Arc<Vec<Box<dyn Continuous>>> = Arc::new(vec![
        Box::new(Normal::new(4., 2.)),
        Box::new(Uniform::new(0., 10.)),
    ]);

    let proposals: Arc<Vec<Box<dyn Continuous>>> = Arc::new(vec![
        Box::new(Normal::new(0., 0.1)),
        Box::new(Normal::new(0., 0.1)),
    ]);

    let n_iter = 3000;

    // let priors: Arc<Vec<Box<dyn Continuous>>> = Arc::new(vec![
    //     Box::new(Normal::new(5., 5.)),
    //     Box::new(Normal::new(5., 5.)),
    //     Box::new(Exponential::new(1. / 15.)),
    // ]);

    // let proposals: Arc<Vec<Box<dyn Continuous>>> = Arc::new(vec![
    //     Box::new(Normal::new(0., 0.5)),
    //     Box::new(Normal::new(0., 0.2)),
    //     Box::new(Normal::new(0., 1.)),
    // ]);

    let n_iter = 2500;

    let chains = (0..num_cpus::get())
        .into_par_iter()
        .map(|i| {
            let now = Instant::now();
            let (samples, acceptance_rates) = sampler::sampler(
                &data,
                n_iter,
                proposal_dist,
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
}
