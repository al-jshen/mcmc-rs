use compute::distributions::*;
use std::sync::Arc;

pub fn sampler(
    xdata: &[f64],
    ydata: &[f64],
    n_iter: usize,
    priors: Arc<Vec<Box<dyn Continuous>>>,
    proposals: Arc<Vec<Box<dyn Continuous>>>,
) -> Vec<Vec<f64>> {
    let mut samples = vec![vec![0.; n_iter]; priors.len()];

    let mut params = priors.iter().map(|d| d.sample()).collect::<Vec<_>>();
    let distmu = create_distmu(&params, xdata);

    let mut lpdf = 0.;
    // lpdf += calc_boxedloglikelihood(&priors, &params);
    lpdf += calc_loglikelihood(&distmu, ydata);

    for n in 1..n_iter {
        for i in 0..proposals.len() {
            let mut new_params = params.clone();
            if i == 2 {
                // only for sigma, restrict to positive numbers
                new_params[i] = loop {
                    let np = new_params[i] + proposals[i].sample();
                    if np > 0. {
                        break np;
                    }
                };
            } else {
                new_params[i] += proposals[i].sample();
            }
            let new_distmu = create_distmu(&new_params, xdata);
            // println!("{:?}", new_distmu);
            let mut new_lpdf = 0.;
            // new_lpdf += calc_boxedloglikelihood(&priors, &new_params);
            new_lpdf += calc_loglikelihood(&new_distmu, ydata);

            // println!("{}", new_lpdf > lpdf);
            // println!("{:?}", distmu);
            // println!("-----------------");
            // eprintln!("{}, {}", lpdf, new_lpdf);
            println!("-------------------------------");
            println!("{:?}, {}", params, lpdf);
            println!("{:?}, {}", new_params, new_lpdf);
            if new_lpdf > lpdf {
                params = new_params;
                lpdf = new_lpdf;
            }
            samples[i][n] = params[i];
        }
    }

    samples
}

fn create_distmu(params: &[f64], xdata: &[f64]) -> Vec<Normal> {
    let mu = xdata
        .iter()
        .map(|x| params[0] + params[1] * x)
        .collect::<Vec<f64>>();
    let distmu = mu
        .iter()
        .map(|x| Normal::new(*x, params[2]))
        .collect::<Vec<_>>();
    distmu
}

fn calc_loglikelihood<T: Continuous>(distrs: &[T], data: &[f64]) -> f64 {
    distrs
        .iter()
        .enumerate()
        .map(|(i, x)| x.pdf(data[i]).ln())
        .sum()
}

fn calc_boxedloglikelihood(distrs: &[Box<dyn Continuous>], data: &[f64]) -> f64 {
    distrs
        .iter()
        .enumerate()
        .map(|(i, x)| x.pdf(data[i]).ln())
        .sum()
}
