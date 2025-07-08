use rayon::prelude::*;

/// Huber Regressor according to: Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics Concomitant scale estimates, p. 172
pub struct HuberRegressor {
    x: Vec<f64>,
    y: Vec<f64>,
    w: Option<f64>,
    b: Option<f64>,
    /// Delta for the Huber loss function
    pub delta: f64,
    /// Tolerance values have to be in to stop the gradient descent
    pub tol: f64,
}

impl HuberRegressor {
    /// Creates a new unfitted regressor
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            w: None,
            b: None,
            delta: 1.35,
            tol: 1e-6,
        }
    }

    /// Fit the regressor to x and y values
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        self.x = x;
        self.y = y;
        let params = self.gradient_descent(2);
        self.w = Some(params[0]);
        self.b = Some(params[1]);
    }

    /// Get the parameters of the previously fitted curve. Returns None if the regressor was not
    /// fitted before. The first value is the slope, the second one the intercept.
    pub fn get_params(&self) -> Option<(f64, f64)> {
        match (self.w, self.b) {
            (Some(w), Some(b)) => Some((w, b)),
            _ => None,
        }
    }

    fn forward_difference(&self, x: &Vec<f64>) -> Vec<f64> {
        const PERTURBATION: f64 = 1e-8;

        let f_x = self.huber_loss(x);

        (0..x.len())
            .into_par_iter()
            .map(|i| {
                let mut x_forward = x.to_owned();
                x_forward[i] += PERTURBATION;
                (self.huber_loss(&x_forward) - f_x) / PERTURBATION
            })
            .collect()
    }

    fn line_search(&self, mut x: Vec<f64>, gradient: Vec<f64>) -> f64 {
        const TAU: f64 = 0.5;
        const C: f64 = 0.5;

        let mut stepsize: f64 = 0.001;
        let m: f64 = gradient.iter().map(|&grad_i| grad_i * grad_i).sum();

        let t = -C * m;
        let f_x = self.huber_loss(&x);

        loop {
            x.iter_mut()
                .zip(gradient.iter())
                .for_each(|(x_i, &grad_i)| *x_i -= stepsize * grad_i);

            if self.huber_loss(&x) <= f_x - stepsize * t {
                return stepsize;
            }

            stepsize *= TAU;
            if stepsize < 1e-10 {
                return stepsize;
            }
        }
    }

    fn gradient_descent(&self, len_of_inputs: usize) -> Vec<f64> {
        let mut x: Vec<f64> = vec![0.0; len_of_inputs];
        let mut grad: Vec<f64> = self.forward_difference(&x);

        let mut abs_sum: f64 = grad.iter().map(|&grad_i| grad_i.abs()).sum();

        while abs_sum >= self.tol {
            let a = self.line_search(x.clone(), grad.clone());
            x.iter_mut()
                .zip(grad.iter())
                .for_each(|(x_i, &grad_i)| *x_i -= a * grad_i);
            grad = self.forward_difference(&x);
            abs_sum = grad.iter().map(|&x| x.abs()).sum();
        }
        x
    }

    fn huber_loss(&self, params: &Vec<f64>) -> f64 {
        let x = &self.x;
        let y = &self.y;
        let w = params[0];
        let b = params[1];
        let residuals: Vec<f64> = y
            .iter()
            .zip(x.iter())
            .map(|(y_i, x_i)| (y_i - (w * x_i + b)).abs())
            .collect();
        residuals
            .iter()
            .map(|r_i| {
                if *r_i <= self.delta {
                    0.5 * r_i * r_i
                } else {
                    self.delta * (r_i - 0.5 * self.delta)
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn within_tolerance(x: f64, y: f64) -> bool {
        const TOL: f64 = 1e-5;
        x - TOL < y && x + TOL > y
    }

    #[test]
    fn test_huber_regressor() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let mut reg = HuberRegressor::new();
        reg.fit(x, y);
        let params = reg.get_params().unwrap();
        assert!(within_tolerance(params.0, 2.0));
        assert!(within_tolerance(params.1, 1.0));
    }
}
