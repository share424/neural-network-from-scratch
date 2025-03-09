use crate::matrix::Matrix;

pub enum Activation {
    RELU,
    SOFTMAX,
}

pub fn apply_activation(x: &Matrix, activation: &Activation) -> Matrix {
    match activation {
        Activation::RELU => relu(&x),
        Activation::SOFTMAX => softmax(&x),
    }
}

pub fn apply_activation_backward(x: &Matrix, activation: &Activation) -> Matrix {
    match activation {
        Activation::RELU => relu_backward(&x),
        Activation::SOFTMAX => softmax_backward(&x),
    }
}

pub fn relu(x: &Matrix) -> Matrix {
    // relu(x) = max(0, x)
    let mut output = Matrix::zeros(x.rows, x.cols);
    for i in 0..x.rows {
        for j in 0..x.cols {
            let value = x.get(i, j);
            if value > 0.0 {
                output.set(i, j, value);
            }
        }
    }

    output
}

pub fn relu_backward(x: &Matrix) -> Matrix {
    // 0 => 0
    // x => 1
    let mut output = Matrix::zeros(x.rows, x.cols);
    for i in 0..x.rows {
        for j in 0..x.cols {
            let value = x.get(i, j);
            if value > 0.0 {
                output.set(i, j, 1.0);
            }
        }
    }

    output
}

pub fn softmax(x: &Matrix) -> Matrix {
    // softmax(x) = exp(x[i]) / sum(exp(x))
    let mut output = Matrix::zeros(x.rows, x.cols);
    for i in 0..x.rows {
        let mut max_value = x.get(0, 0);
        for j in 0..x.cols {
            if x.get(i, j) > max_value {
                max_value = x.get(i, j);
            }
        }

        let mut sum = 0.0;
        let mut temp_data = vec![0.0; x.cols];
        for j in 0..x.cols {
            temp_data[j] = (x.get(i, j) - max_value).exp();
            sum += temp_data[j];
        }

        for j in 0..x.cols {
            let exp_j = temp_data[j];
            let value = exp_j / sum;
            output.set(i, j, value);
        }
    }

    output
}

pub fn softmax_backward(x: &Matrix) -> Matrix {
    let data = vec![1.0; x.rows * x.cols];
    Matrix {
        data,
        rows: x.rows,
        cols: x.cols,
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    use super::{relu, softmax};

    #[test]
    pub fn test_relu() {
        let x = Matrix::new(vec![0.0, -1.0, 1.0, 2.0], 2, 2);
        let output = relu(&x);

        let expected = vec![0.0, 0.0, 1.0, 2.0];
        for i in 0..2 * 2 {
            assert!(output.data[i] == expected[i]);
        }
    }

    #[test]
    pub fn test_softmax() {
        let x = Matrix::new(vec![1.0, 1.0], 1, 2);
        let output = softmax(&x);

        let expected = vec![0.5, 0.5];
        for i in 0..2 {
            assert!(
                output.data[i] == expected[i],
                "{} != {}",
                output.data[i],
                expected[i]
            );
        }
    }
}
