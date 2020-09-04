#[macro_use]
mod sampler;
mod utils;
use rayon::prelude::*;
use statistics::distributions::*;
use std::sync::Arc;
use std::time::Instant;

fn main() {
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
