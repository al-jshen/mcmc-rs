// use crate::distr_wrapper;
// use rand_distr::Distribution;
// use statrs::distribution::Continuous;
use statistics::distributions::*;

pub fn sampler<T>(
    data: &[f64],
    n_iter: usize,
    distr_fn: T,
    distr_priors: Vec<Box<dyn Continuous>>,
    distr_proposals: Vec<Box<dyn Continuous>>,
) -> (Vec<Vec<f64>>, Vec<usize>)
where
    T: Continuous + Clone + Copy,
{
    let mut acceptances: Vec<usize> = vec![0; distr_priors.len()];

    let mut params_current = distr_priors.iter().map(|d| d.sample()).collect::<Vec<_>>();
    let mut samples = vec![vec![0.; n_iter]; distr_priors.len()];

    for (i, v) in params_current.iter().enumerate() {
        samples[i][0] = *v;
    }

    for i in 1..n_iter {
        for j in 0..distr_priors.len() {
            let mut params_proposed = params_current.clone();
            params_proposed[j] += distr_proposals[j].sample();

            let mut distr_current = distr_fn;
            distr_current.update(&params_current);
            let mut distr_proposed = distr_current.clone();
            distr_proposed.update(&params_proposed);

            let accept = accept_reject(
                &distr_current,
                &distr_proposed,
                params_current[j],
                params_proposed[j],
                &distr_priors[j],
                data,
            );
            if accept {
                params_current[j] = params_proposed[j];
                acceptances[j] += 1;
            }
            samples[j][i] = params_current[j];
        }
    }

    (samples, acceptances)
}

pub fn accept_reject<T: Continuous>(
    distr_current: &T,
    distr_proposal: &T,
    param_current: f64,
    param_proposal: f64,
    distr_prior: &Box<dyn Continuous>,
    data: &[f64],
) -> bool {
    let likelihood_current = calc_loglikelihood(distr_current, data);
    let likelihood_proposal = calc_loglikelihood(distr_proposal, data);

    let prior_current = distr_prior.pdf(param_current).ln();
    let prior_proposal = distr_prior.pdf(param_proposal).ln();

    let p_current = likelihood_current + prior_current;
    let p_proposal = likelihood_proposal + prior_proposal;

    let p_accept = f64::min((p_proposal - p_current).exp(), 1.);
    fastrand::f64() < p_accept
}

pub fn calc_loglikelihood<T: Continuous>(distr: &T, data: &[f64]) -> f64 {
    data.iter().map(|x| distr.pdf(*x).ln()).sum()
}
