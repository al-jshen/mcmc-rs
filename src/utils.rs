pub fn scale(data: Vec<f64>, low: f64, high: f64) -> Vec<f64> {
    data.iter()
        .map(|x| {
            (high - low) * (x - data.iter().cloned().fold(f64::INFINITY, f64::min))
                / (data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    - data.iter().cloned().fold(f64::INFINITY, f64::min))
                + low
        })
        .collect::<Vec<_>>()
}
