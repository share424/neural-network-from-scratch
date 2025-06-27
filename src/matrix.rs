use std::fmt;
use std::simd::f32x8;
use std::simd::num::SimdFloat;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:8.4} ", self.get(i, j))?;
            }
            writeln!(f, "")?
        }
        Ok(())
    }
}

impl Matrix {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert!(data.len() == rows * cols);

        Matrix { data, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols];

        Matrix { data, rows, cols }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows);
        assert!(col < self.cols);

        let index = self.cols * row + col;

        self.data[index]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        assert!(row < self.rows);
        assert!(col < self.cols);

        let index = self.cols * row + col;

        self.data[index] = value;
    }

    pub fn random(rows: usize, cols: usize, low: f32, high: f32) -> Self {
        let mut data = vec![0.0; rows * cols];

        for i in 0..rows * cols {
            let rnd = rand::random::<u32>();
            data[i] = low + (rnd as f32 / u32::MAX as f32) * (high - low);
        }

        Matrix { data, rows, cols }
    }

    pub fn transpose(&self) -> Matrix {
        let mut output = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.get(i, j);
                output.set(j, i, value);
            }
        }

        output
    }
}

pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.cols == b.rows);

    let b_t = b.transpose();
    let mut matrix = Matrix::zeros(a.rows, b.cols);

    for i in 0..a.rows {
        for j in 0..b.cols {
            let a_row = &a.data[i * a.cols..(i + 1) * a.cols];
            let b_t_row = &b_t.data[j * b_t.cols..(j + 1) * b_t.cols];

            let mut sum_vec = f32x8::splat(0.0); // 0 0 0 0 0 0 0 0
            let mut k = 0;
            let len = a.cols;
            let chunks = len / 8;

            for _ in 0..chunks {
                let a_vec = f32x8::from_slice(&a_row[k..k + 8]);
                let b_vec = f32x8::from_slice(&b_t_row[k..k + 8]);
                sum_vec = (a_vec * b_vec) + sum_vec;
                k += 8;
            }
            let mut sum = sum_vec.reduce_sum();

            for l in k..len {
                sum += a_row[l] * b_t_row[l];
            }

            matrix.set(i, j, sum);
        }
    }

    return matrix;
}

pub fn matmul_elementwise(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.rows == b.rows);
    assert!(a.cols == b.cols);

    let mut output = Matrix::zeros(a.rows, a.cols);

    for i in 0..a.rows {
        for j in 0..a.cols {
            let value_a = a.get(i, j);
            let value_b = b.get(i, j);
            output.set(i, j, value_a * value_b);
        }
    }

    output
}

pub fn matadd(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.rows == b.rows);
    assert!(a.cols == b.cols);

    let mut output = Matrix::zeros(a.rows, a.cols);

    for i in 0..a.rows {
        for j in 0..a.cols {
            let value_a = a.get(i, j);
            let value_b = b.get(i, j);
            output.set(i, j, value_a + value_b);
        }
    }

    output
}

pub fn matmul_scalar(a: &Matrix, scalar: f32) -> Matrix {
    let mut output = Matrix::zeros(a.rows, a.cols);

    for i in 0..a.rows {
        for j in 0..a.cols {
            let value_a = a.get(i, j);
            output.set(i, j, value_a * scalar);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::{Matrix, matmul};

    #[test]
    fn test_matmul() {
        let a = Matrix::new(vec![1.0, 1.0, 0.0, 2.0], 2, 2);
        let b = Matrix::new(vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0], 2, 3);

        let expected = vec![0.0, 1.0, 2.0, 0.0, 0.0, 2.0];

        let c = matmul(&a, &b);
        for i in 0..2 * 3 {
            assert!(c.data[i] == expected[i]);
        }
    }
}
