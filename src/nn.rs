use crate::{
    activation::{Activation, apply_activation, apply_activation_backward},
    matrix::{Matrix, matmul, matmul_elementwise},
};

pub struct Linear {
    pub weight: Matrix,
    pub bias: Matrix,
    pub activation: Activation,
    pub input: Option<Matrix>,
    pub z: Option<Matrix>,
    pub dw: Option<Matrix>,
    pub db: Option<Matrix>,
}

impl Linear {
    pub fn new(input: usize, output: usize, activation: Activation) -> Self {
        let weight = Matrix::random(input, output, -1.0, 1.0);
        let bias = Matrix::zeros(1, output);

        Linear {
            weight,
            bias,
            activation,
            input: None,
            z: None,
            dw: None,
            db: None,
        }
    }

    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        // y = xW + b
        self.input = Some(x.clone()); // save for backward
        let mut output = matmul(&x, &self.weight);

        for i in 0..output.rows {
            for j in 0..self.bias.cols {
                let value = output.get(i, j);
                let bias = self.bias.get(0, j);
                output.set(i, j, value + bias);
            }
        }

        // apply activation
        self.z = Some(output.clone());
        output = apply_activation(&output, &self.activation);

        output
    }

    pub fn backward(&mut self, gradient: &Matrix) -> Matrix {
        // 1. calculate dy/dz
        let act_prime = apply_activation_backward(&self.z.as_ref().unwrap(), &self.activation);
        let gradient = matmul_elementwise(&gradient, &act_prime);
        // gradient = dL/dy * dy/dz

        // 2. calculate dL/dw
        // y = xW + b
        // dy/dw = x
        // dL/dW = dL/dy * dy/dz * dy/dW
        self.dw = Some(matmul(&self.input.as_ref().unwrap().transpose(), &gradient));

        // 3. calculate dL/db
        // y = xW + b
        // dy/db = 1
        // dL/db = dL/dy * dy/dz * dy/db
        self.db = Some(gradient.clone());

        let gradient = matmul(&gradient, &self.weight.transpose());

        gradient
    }

    pub fn get_total_params(&self) -> usize {
        self.weight.rows * self.weight.cols * self.bias.cols
    }

    pub fn export(&self) -> Vec<f32> {
        // concat W and b
        let mut data = vec![0.0; self.weight.rows * self.weight.cols * self.bias.cols];

        let mut idx = 0;
        for i in 0..self.weight.data.len() {
            data[idx] = self.weight.data[i];
            idx += 1;
        }

        for i in 0..self.bias.data.len() {
            data[idx] = self.bias.data[i];
            idx += 1;
        }

        data
    }

    pub fn load(&mut self, data: &Vec<f32>) {
        let mut idx = 0;
        for i in 0..self.weight.data.len() {
            self.weight.data[i] = data[idx];
            idx += 1;
        }

        for i in 0..self.bias.data.len() {
            self.bias.data[i] = data[idx];
            idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    use super::Linear;

    #[test]
    pub fn test_forward() {
        let mut layer = Linear::new(3, 2, crate::activation::Activation::RELU);
        let x = Matrix::random(2, 3, -1.0, 1.0);
        let output = layer.forward(&x);

        assert!(output.rows == 2);
        assert!(output.cols == 2);
    }
}
