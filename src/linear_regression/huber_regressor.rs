use argmin::{
    core::{observers::ObserverMode, CostFunction, Executor, Gradient, State},
    solver::{linesearch::MoreThuenteLineSearch, quasinewton::LBFGS},
};
use argmin_observer_slog::SlogLogger;
use ndarray::{array, Array1, Zip};

/// Huber Regressor according to: Peter J. Huber, Elvezio M. Ronchetti, Robust Statistics Concomitant scale estimates, p. 172
pub struct HuberRegressor {
    x: Array1<f64>,
    y: Array1<f64>,
    /// Delta for the Huber loss function
    pub delta: f64,
    /// Tolerance values have to be in to stop the gradient descent
    pub tol: f64,
    pub max_iter: usize,
}

impl HuberRegressor {
    /// Creates a new unfitted regressor
    pub fn from(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self {
            x: Array1::from_vec(x),
            y: Array1::from_vec(y),
            delta: 1.35,
            tol: 1e-6,
            max_iter: 2000,
        }
    }

    fn huber_loss(&self, params: &Array1<f64>) -> f64 {
        let y_hat = &self.x * params[[0]] + params[[1]];
        let residuals = &self.y - y_hat;
        residuals
            .mapv(|r_i| {
                if r_i.abs() <= self.delta {
                    0.5 * r_i * r_i
                } else {
                    self.delta * (r_i.abs() - 0.5 * self.delta)
                }
            })
            .sum()
    }
}

impl CostFunction for HuberRegressor {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(self.huber_loss(param))
    }
}

impl Gradient for HuberRegressor {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let y_hat = &self.x * param[[0]] + param[[1]];
        let residuals = &self.y - y_hat;
        let mut w_values = Array1::zeros(residuals.len());
        Zip::from(&mut w_values)
            .and(&residuals)
            .and(&self.x)
            .for_each(|w_value, r, x| {
                *w_value = if r.abs() <= self.delta {
                    -x * r
                } else {
                    -self.delta * x * r.signum()
                }
            });
        let mut b_values = Array1::zeros(residuals.len());
        Zip::from(&mut b_values)
            .and(&residuals)
            .for_each(|w_value, r| {
                *w_value = if r.abs() <= self.delta {
                    -r
                } else {
                    -self.delta * r.signum()
                }
            });
        let w = w_values.sum();
        let b = b_values.sum();
        // eprintln!(
        //     "{}, r: {}, w_s: {}, b_s: {}, w: {}, b: {}",
        //     param, residuals, w_values, b_values, w, b
        // );
        Ok(array![w, b])
    }
}

/// Fit the regressor to x and y values
pub fn solve(cost: HuberRegressor) -> Vec<f64> {
    // Define initial parameter vector
    let init_param: Array1<f64> = array![0.0, 0.0];

    // set up a line search
    let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();

    // Set up solver
    let solver = LBFGS::new(linesearch, 7);

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        // .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();

    // Print result

    let best = res.state().get_best_param().unwrap();
    best.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn within_tolerance(x: f64, y: f64) -> bool {
        const TOL: f64 = 1e-2;
        x - TOL < y && x + TOL > y
    }

    #[test]
    fn test_huber_regressor() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let hub = HuberRegressor::from(x, y);
        let res = solve(hub);
        assert!(within_tolerance(res[0], 2.0));
        assert!(within_tolerance(res[1], 1.0));
    }
}
