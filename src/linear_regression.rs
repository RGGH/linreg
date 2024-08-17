#[derive(Debug, Clone)]
pub struct LinearRegression {
    pub coefficients: Vec<f64>,
}

impl LinearRegression {
    pub fn predict(&self, inputs: &[f64]) -> f64 {
        self.coefficients
            .iter()
            .zip(inputs)
            .map(|(w, x)| w * x)
            .sum()
    }
}
