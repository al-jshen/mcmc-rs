use rayon::prelude::*;

pub fn scale(data: Vec<f64>, low: f64, high: f64) -> Vec<f64> {
    data.par_iter()
        .map(|x| (high - low) * (x - min(&data)) / (max(&data) - min(&data)) + low)
        .collect::<Vec<_>>()
}

pub fn standardize(data: Vec<f64>) -> Vec<f64> {
    let xbar = mean(&data);
    let sigma = variance(&data).sqrt();
    data.par_iter().map(|x| {
        (x - xbar) / sigma
    }).collect::<Vec<f64>>()
}

fn max<T>(data: &[T]) -> T
where
    T: std::cmp::PartialOrd + Copy,
{
    *(data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap())
}

fn min<T>(data: &[T]) -> T
where
    T: std::cmp::PartialOrd + Copy,
{
    *(data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap())
}

fn mean(data: &[f64]) -> f64 {
    data.par_iter().sum::<f64>() / data.len() as f64
}

fn variance(data: &[f64]) -> f64 {
    // var = mean(abs(x - x.mean())**2)
    let xbar = mean(data);
    mean(
        &data.par_iter().map(|x| {
            (x - xbar).abs().powi(2)
        }).collect::<Vec<f64>>()
    )
}
