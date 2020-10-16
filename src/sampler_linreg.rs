use compute::distributions::*;
use std::sync::Arc;

#[allow(dead_code)]
fn sampler(
    xdata: &[f64],
    ydata: &[f64],
    n_iter: usize,
    distalpha: Normal,      // distribution of alpha param
    distbeta: Normal,       // distribution of beta param
    distsigma: Exponential, // distribution of sigma param
    propalpha: Normal,      // how to draw proposals to change alpha param
    propbeta: Normal,       // how to draw proposals to change beta param
    propsigma: Normal,      // how to draw proposals to change sigma param
) {
    let alpha = distalpha.sample();
    let beta = distbeta.sample();
    let sigma = distsigma.sample();
    let mu = &xdata.iter().map(|x| alpha + beta * x).collect::<Vec<f64>>();
    let distmu = &mu
        .iter()
        .map(|x| Normal::new(*x, sigma))
        .collect::<Vec<_>>();
    // how good is distmu?
    // create some evaluation method.
    let mut lpdf = 0.;
    lpdf += distalpha.pdf(alpha).ln();
    lpdf += distbeta.pdf(beta).ln();
    lpdf += distsigma.pdf(sigma).ln();
    lpdf += calc_loglikelihood(distmu, ydata);

    // change the alpha first, see if that makes things better.

    let alpha2 = distalpha.sample();
    let mu2 = &xdata
        .iter()
        .map(|x| alpha2 + beta * x)
        .collect::<Vec<f64>>();
    let distmu2 = &mu2
        .iter()
        .map(|x| Normal::new(*x, sigma))
        .collect::<Vec<_>>();

    let mut lpdf2 = 0.;
    lpdf2 += distalpha.pdf(alpha2);
    lpdf2 += distbeta.pdf(beta).ln();
    lpdf += distsigma.pdf(sigma).ln();
    lpdf += calc_loglikelihood(distmu2, ydata);

    if (lpdf2 > lpdf) {
        // use alpha2 instead of alpha1
    } else {
        // throw away alpha2
    }
    // then add alpha1 or alpha2 to the samples for alpha

    // then change the beta, see if that makes things better.
    ()
}

fn calc_loglikelihood<T: Continuous>(mean_dists: &[T], data: &[f64]) -> f64 {
    mean_dists
        .iter()
        .enumerate()
        .map(|(i, x)| x.pdf(data[i]).ln())
        .sum()
}
