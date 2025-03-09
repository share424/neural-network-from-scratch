use std::{
    fs::{File, OpenOptions},
    io::{self, BufReader, Read, Write},
};

use rand::{Rng, distr::Uniform};

use crate::{
    matrix::{Matrix, matadd, matmul_scalar},
    nn::Linear,
};

pub struct Model {
    pub layers: Vec<Linear>,
}

impl Model {
    pub fn new(layers: Vec<Linear>) -> Self {
        Model { layers }
    }

    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let mut x = x.clone();
        for i in 0..self.layers.len() {
            x = self.layers[i].forward(&x);
        }

        x
    }

    pub fn train(
        &mut self,
        dataset: &Matrix,
        labels: &Matrix,
        lr: f32,
        epoch: usize,
        batch_size: usize,
    ) {
        let mut rng = rand::rng();
        for ep in 0..epoch {
            println!("Epoch: {ep}");

            for i in 0..dataset.rows / batch_size {
                let mut data = vec![0.0; dataset.cols * batch_size];
                let mut label = vec![0.0; labels.cols * batch_size];

                for b in 0..batch_size {
                    let range = Uniform::new(0, dataset.rows).expect("Error");
                    let rnd_idx = rng.sample(range);
                    for j in 0..dataset.cols {
                        data[dataset.cols * b + j] = dataset.get(rnd_idx, j) / 255.0;
                    }

                    for j in 0..labels.cols {
                        label[labels.cols * b + j] = labels.get(rnd_idx, j);
                    }
                }

                let x = Matrix::new(data, batch_size, dataset.cols);
                let y = Matrix::new(label, batch_size, labels.cols);

                let output = self.forward(&x);
                let loss = self.calculate_loss(&output, &y);

                println!("Loss: {loss}");

                // backpropagation

                // 1. calculate dL/dy
                // because we use cross-entropy,
                // dL/dy = y_pred - y_true
                let mut gradient = matadd(&output, &matmul_scalar(&y, -1.0));

                for l in (0..self.layers.len()).rev() {
                    gradient = self.layers[l].backward(&gradient);

                    // update weight, SGD
                    // gradient descent = w - gradient * lr
                    let delta = matmul_scalar(&self.layers[l].dw.as_ref().unwrap(), -lr);
                    self.layers[l].weight = matadd(&self.layers[l].weight, &delta);

                    // update bias
                    // gradient descent = b - gradient * lr
                    let delta = matmul_scalar(&self.layers[l].db.as_ref().unwrap(), -lr);
                    for bi in 0..delta.rows {
                        for bj in 0..delta.cols {
                            self.layers[l].bias.data[bj] += delta.get(bi, bj);
                        }
                    }
                }
            }
        }
    }

    pub fn calculate_loss(&self, output: &Matrix, labels: &Matrix) -> f32 {
        // log-loss / cross-entropy
        // -1 * sum(p(x) * log(q))
        // p = labels
        // q = predictions
        assert!(output.rows == labels.rows);

        let mut loss = 0.0;
        let eps: f32 = 1e-6 as f32;
        for i in 0..labels.rows {
            for j in 0..labels.cols {
                let output_clamped = output.get(i, j).max(eps).min(1.0 - eps);
                let log_q = output_clamped.ln();
                let p = labels.get(i, j);
                loss += p * log_q;
            }
        }

        (-1.0 * loss) / (output.rows as f32)
    }

    pub fn evaluate(&mut self, dataset: &Matrix, labels: &Matrix) -> f32 {
        let mut acc: i32 = 0;
        let label_indices = argmax(&labels);
        for i in 0..dataset.rows {
            let mut x = Matrix::zeros(1, dataset.cols);

            for j in 0..dataset.cols {
                x.set(0, j, dataset.get(i, j));
            }

            let output = self.forward(&x);

            let y_pred = argmax(&output);

            if y_pred[0] == label_indices[i] {
                acc += 1;
            }
        }

        acc as f32 / dataset.rows as f32
    }

    pub fn save(&self, filepath: &str) {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filepath)
            .unwrap();

        for layer in self.layers.iter() {
            let data = layer.export();

            let bytes = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<f32>(),
                )
            };

            file.write_all(bytes).unwrap();
        }
    }

    pub fn load(&mut self, filepath: &str) {
        let file = File::open(filepath).unwrap();
        let mut reader = BufReader::new(file);

        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).unwrap();

        let mut current_layer = 0;
        let mut total_params = self.layers[current_layer].get_total_params();
        let mut data = vec![0.0; total_params];

        let mut offset = 0;
        for chunk in buffer.chunks(4) {
            if chunk.len() == 4 {
                let value = f32::from_le_bytes(chunk.try_into().unwrap());
                data[offset] = value;
                offset += 1;
            }

            if offset >= total_params {
                self.layers[current_layer].load(&data);
                offset = 0;
                current_layer += 1;
                if current_layer >= self.layers.len() {
                    continue;
                }
                total_params = self.layers[current_layer].get_total_params();
                data = vec![0.0; total_params];
            }
        }
    }
}

pub fn argmax(data: &Matrix) -> Vec<usize> {
    let mut max_indices = vec![0; data.rows];
    for i in 0..data.rows {
        let mut max_idx = 0;
        for j in 0..data.cols {
            if data.get(i, j) > data.get(i, max_idx) {
                max_idx = j
            }
        }
        max_indices[i] = max_idx
    }

    max_indices
}
