use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{Dyn, Matrix, OMatrix, Owned, VecStorage, U1, U3};

pub struct PowerLawIntercept {
    x: Vec<f64>,
    y: Vec<f64>,
    params: Vec<f64>,
}

impl LeastSquaresProblem<f64, Dyn, U3> for PowerLawIntercept {
    type ParameterStorage = Owned<f64, U3, U1>;
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = Owned<f64, Dyn, U3>;

    fn set_params(&mut self, x: &nalgebra::Vector<f64, U3, Self::ParameterStorage>) {
        self.params = x.iter().copied().collect();
    }

    fn params(&self) -> nalgebra::Vector<f64, U3, Self::ParameterStorage> {
        nalgebra::Vector3::new(self.params[0], self.params[1], self.params[2])
    }

    fn residuals(&self) -> Option<Matrix<f64, Dyn, U1, Self::ResidualStorage>> {
        let k = self.params[0];
        let alpha = self.params[1];
        let c = self.params[2];

        let mut r = Matrix::<f64, Dyn, U1, Self::ResidualStorage>::zeros(self.x.len());
        for i in 0..self.x.len() {
            r[i] = k * self.x[i].powf(-alpha) + c - self.y[i];
        }
        Some(r)
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, U3>> {
        let k = self.params[0];
        let alpha = self.params[1];

        let mut j = OMatrix::<f64, Dyn, U3>::zeros(self.x.len());
        for i in 0..self.x.len() {
            let x = self.x[i];
            j[(i, 0)] = x.powf(-alpha); // ∂f/∂K
            j[(i, 1)] = -k * x.powf(-alpha) * x.ln(); // ∂f/∂alpha
            j[(i, 2)] = 1.0; // ∂f/∂c
        }
        Some(j)
    }
}

impl PowerLawIntercept {
    pub fn from(x: Vec<f64>, y: Vec<f64>) -> Self {
        PowerLawIntercept {
            x,
            y,
            params: vec![1.0, 1.0, 0.0],
        }
    }

    pub fn get_params(&self) -> Vec<f64> {
        self.params.clone()
    }
}

pub fn solve(power_law: PowerLawIntercept) -> Vec<f64> {
    let (result, report) = LevenbergMarquardt::new().minimize(power_law);
    if !report.termination.was_successful() {
        eprintln!("Levenberg-Marquardt terminated: {:?}", report.termination);
    }
    // assert!(report.termination.was_successful());
    // assert!(report.objective_function.abs() < 1e-10);
    result.get_params()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn within_tolerance(x: f64, y: f64) -> bool {
        const TOL: f64 = 1e-2;
        x - TOL < y && x + TOL > y
    }

    #[test]
    fn test_power_law_intercept() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![7.5, 4.875, 4.3888889, 4.21875, 4.14, 4.0972222];
        let pl = PowerLawIntercept::from(x, y);
        let res = solve(pl);
        eprintln!("Params: {:?}", res);
        assert!(within_tolerance(res[0], 3.5));
        assert!(within_tolerance(res[1], 2.0));
        assert!(within_tolerance(res[2], 4.0));
    }
}
