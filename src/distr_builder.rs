use rand::distributions::Distribution;

pub struct DistrBuilder {
    params: Vec<f64>,
    dist: dyn Distribution<f64>,
}

impl DistrBuilder {
    pub fn new(dist: impl Distribution<f64>) -> Self {
        DistrBuilder {
            dist: dist,
            params: Vec::with_capacity(3),
        }
    }
}
