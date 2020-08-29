mod distr_wrapper;
mod utils;
use rand::prelude::*;
use statrs::distribution::*;
// use std::{
//     fs::File,
//     io::{BufRead, BufReader},
//     str::FromStr,
// };
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let mut rng = rand::thread_rng();
    let n = 1000;
    let data = Normal::new(5., 2.)
        .unwrap()
        .sample_iter(&mut rng)
        .take(n)
        .collect::<Vec<f64>>();

    // let dat_file = File::open("test_beta.csv").unwrap();
    // let buf = BufReader::new(dat_file);
    // let mut data = buf.lines()
    //     .map(|l| f64::from_str(&l.unwrap()).unwrap())
    //     .collect::<Vec<f64>>();

    // data = utils::scale(utils::standardize(data), 0., 1.);

    let distr_fn = distr_wrapper::DNormal;

    let prior_mu = Uniform::new(0., 10.).unwrap();
    let prior_sigma = LogNormal::new(0., 1.).unwrap();

    let proposal_mu = Normal::new(0., 0.1).unwrap();
    let proposal_sigma = Normal::new(0., 0.1).unwrap();

    let n_iter = 3000;

    let chains = (0..num_cpus::get())
        .into_par_iter()
        .map(|i| {
            let now = Instant::now();
            let (mu, sigma, mu_accept, sigma_accept) = sampler(
                &data,
                n_iter as usize,
                distr_fn,
                prior_mu,
                prior_sigma,
                proposal_mu,
                proposal_sigma,
            );
            eprintln!(
                "chain: {} \t time: {:.3}s \t mu accept rate: {:.3} \t sigma accept rate: {:.3}",
                i,
                now.elapsed().as_secs_f64(),
                mu_accept as f64 / n_iter as f64,
                sigma_accept as f64 / n_iter as f64
            );
            (mu, sigma)
        })
        .collect::<Vec<(Vec<f64>, Vec<f64>)>>();

    println!("d={:?}", data);
    //data.iter().for_each(|x| println!("{}", x));
    // mu_samples.iter().for_each(|x| println!("{}", x));
    // sigma_samples.iter().for_each(|x| println!("{}", x));
    println!("chains={:?}", chains);
}

fn sampler<T, V, W, X, Y>(
    data: &[f64],
    n_iter: usize,
    distr_fn: T,
    prior_mu: V,
    prior_sigma: W,
    proposal_mu: X,
    proposal_sigma: Y,
) -> (Vec<f64>, Vec<f64>, usize, usize)
where
    T: distr_wrapper::DWrapper + Copy,
    V: Distribution<f64> + Continuous<f64, f64>,
    W: Distribution<f64> + Continuous<f64, f64>,
    X: Distribution<f64>,
    Y: Distribution<f64>,
{
    let mut mu_accepts = 0;
    let mut sigma_accepts = 0;
    let mut rng = thread_rng();

    let mut mu_current = prior_mu.sample(&mut rng);
    let mut sigma_current = prior_sigma.sample(&mut rng);
    let mut mu_samples: Vec<f64> = vec![0.; n_iter];
    let mut sigma_samples: Vec<f64> = vec![0.; n_iter];
    mu_samples[0] = mu_current;
    sigma_samples[0] = sigma_current;


    for i in 1..n_iter {
        let mut mu_proposal = mu_current + proposal_mu.sample(&mut rng);
        // while mu_proposal <= 0. {
        //     // println!("mu_current={}, mu_proposal={}", mu_current, mu_proposal);
        //     mu_proposal = mu_current + proposal_mu.sample(&mut rng);
        // }

        let distr_current = distr_fn.new(mu_current, sigma_current);
        let distr_proposal = distr_fn.new(mu_proposal, sigma_current);

        let accept = accept_reject(
            &distr_current,
            &distr_proposal,
            mu_current,
            mu_proposal,
            &prior_mu,
            data,
            &mut rng,
        );

        if accept {
            mu_current = mu_proposal;
            mu_accepts += 1;
        }

        mu_samples[i] = mu_current;

        //

        let mut sigma_proposal = sigma_current + proposal_sigma.sample(&mut rng);
        while sigma_proposal <= 0. {
            // println!("mu_current={}, mu_proposal={}", mu_current, mu_proposal);
            sigma_proposal = sigma_current + proposal_sigma.sample(&mut rng);
        }

        let distr_current = distr_fn.new(mu_current, sigma_current);
        let distr_proposal = distr_fn.new(mu_current, sigma_proposal);

        let accept = accept_reject(
            &distr_current,
            &distr_proposal,
            sigma_current,
            sigma_proposal,
            &prior_sigma,
            data,
            &mut rng,
        );

        if accept {
            sigma_current = sigma_proposal;
            sigma_accepts += 1;
        }

        sigma_samples[i] = sigma_current;
    }

    (mu_samples, sigma_samples, mu_accepts, sigma_accepts)
}

fn accept_reject<T: Continuous<f64, f64>, U: Continuous<f64, f64>>(
    distr_current: &T,
    distr_proposal: &T,
    param_current: f64,
    param_proposal: f64,
    distr_prior: &U,
    data: &[f64],
    mut rng: impl Rng,
) -> bool {
    let likelihood_current = calc_loglikelihood(distr_current, data);
    let likelihood_proposal = calc_loglikelihood(distr_proposal, data);

    let prior_current = distr_prior.pdf(param_current).ln();
    let prior_proposal = distr_prior.pdf(param_proposal).ln();

    let p_current = likelihood_current + prior_current;
    let p_proposal = likelihood_proposal + prior_proposal;

    let p_accept = f64::min((p_proposal - p_current).exp(), 1.);
    rng.gen_bool(p_accept)
}

fn calc_loglikelihood<T: Continuous<f64, f64>>(distr: &T, data: &[f64]) -> f64 {
    data.iter().map(|x| distr.pdf(*x).ln()).sum()
}
