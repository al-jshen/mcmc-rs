use compute::distributions::*;
use std::sync::Arc;

#[allow(dead_code)]
fn sampler(
    xdata: &[f64],
    ydata: &[f64],
    n_iter: usize,
    distalpha: Normal,
    distbeta: Normal,
    sigma: f64,
) {
    let alpha = distalpha.sample();
    let beta = distbeta.sample();
    let mu = &xdata.iter().map(|x| alpha + beta * x).collect::<Vec<f64>>();
    let distmu = &mu
        .iter()
        .map(|x| Normal::new(*x, sigma))
        .collect::<Vec<_>>();
    ()
}

fn calc_loglikelihood<T: Continuous>(mean_dists: &[T], data: &[f64]) -> f64 {
    mean_dists
        .iter()
        .enumerate()
        .map(|(i, x)| x.pdf(data[i]).ln())
        .sum()
}
