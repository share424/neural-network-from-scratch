pub mod activation;
pub mod matrix;
pub mod model;
pub mod nn;

use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::process;

use crate::activation::Activation;
use crate::matrix::Matrix;
use crate::model::Model;
use crate::nn::Linear;

fn load_mnist_as_matrix(filepath: &str) -> io::Result<Matrix> {
    let file = File::open(filepath)?;
    let mut reader = BufReader::new(file);

    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    // calculate number of images (total bytes / (784 * 4 bytes per f32))
    let num_images = buffer.len() / (784 * 4);
    let mut data = Vec::with_capacity(num_images * 784);

    for chunk in buffer.chunks(4) {
        if chunk.len() == 4 {
            let value = f32::from_le_bytes(chunk.try_into().unwrap());
            data.push(value);
        }
    }

    Ok(Matrix::new(data, num_images, 784))
}

fn load_mnist_labels_as_matrix(filepath: &str) -> io::Result<Matrix> {
    let file = File::open(filepath)?;
    let mut reader = BufReader::new(file);

    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    const NUM_CLASSES: usize = 10;
    let num_labels = buffer.len() / 4;

    // convert to one hot encoding
    // 3 => [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    let mut data = vec![0.0; num_labels * NUM_CLASSES];

    for (i, chunk) in buffer.chunks(4).enumerate() {
        let label = f32::from_le_bytes(chunk.try_into().unwrap()) as usize;
        if label < NUM_CLASSES {
            data[i * NUM_CLASSES + label] = 1.0;
        }
    }

    Ok(Matrix::new(data, num_labels, NUM_CLASSES))
}

fn print_usage() {
    println!("Neural Network CLI");
    println!("Usage:");
    println!(
        "  Train mode: ./main train --train-data <x_train.bin> --label <y_train.bin> [--output <model.bin>] [--lr <learning_rate>] [--epoch <num_epochs>] [--batch-size <batch_size>]"
    );
    println!(
        "  Evaluate mode: ./main evaluate --test-data <x_test.bin> --label <y_test.bin> [--model <model.bin>]"
    );
}

fn train(
    train_data_path: &str,
    label_path: &str,
    output_path: &str,
    lr: f32,
    epoch: usize,
    batch_size: usize,
) -> io::Result<()> {
    println!("Loading training data from {}...", train_data_path);
    let x_train = load_mnist_as_matrix(train_data_path)?;

    println!("Loading training labels from {}...", label_path);
    let y_train = load_mnist_labels_as_matrix(label_path)?;

    println!("Creating model...");
    let layers = vec![
        Linear::new(784, 128, Activation::RELU),
        Linear::new(128, 10, Activation::SOFTMAX),
    ];
    let mut model = Model::new(layers);

    println!(
        "Training model with learning rate: {}, epochs: {}, batch size: {}...",
        lr, epoch, batch_size
    );
    model.train(&x_train, &y_train, lr, epoch, batch_size);

    println!("Saving model to {}...", output_path);
    model.save(output_path);

    println!("Training complete!");
    Ok(())
}

fn evaluate(test_data_path: &str, label_path: &str, model_path: &str) -> io::Result<()> {
    println!("Loading test data from {}...", test_data_path);
    let x_test = load_mnist_as_matrix(test_data_path)?;

    println!("Loading test labels from {}...", label_path);
    let y_test = load_mnist_labels_as_matrix(label_path)?;

    println!("Loading model from {}...", model_path);
    let layers = vec![
        Linear::new(784, 128, Activation::RELU),
        Linear::new(128, 10, Activation::SOFTMAX),
    ];
    let mut model = Model::new(layers);
    model.load(model_path);

    println!("Evaluating model...");
    let accuracy = model.evaluate(&x_test, &y_test);

    println!("Test Accuracy: {}", accuracy);
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "train" => {
            let mut train_data_path = "";
            let mut label_path = "";
            let mut output_path = "model.bin";
            let mut lr = 0.003;
            let mut epoch = 1;
            let mut batch_size = 128;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--train-data" => {
                        if i + 1 < args.len() {
                            train_data_path = &args[i + 1];
                            i += 2;
                        } else {
                            println!("Error: Missing value for --train-data");
                            process::exit(1);
                        }
                    }
                    "--label" => {
                        if i + 1 < args.len() {
                            label_path = &args[i + 1];
                            i += 2;
                        } else {
                            println!("Error: Missing value for --label");
                            process::exit(1);
                        }
                    }
                    "--output" => {
                        if i + 1 < args.len() {
                            output_path = &args[i + 1];
                            i += 2;
                        } else {
                            println!("Error: Missing value for --output");
                            process::exit(1);
                        }
                    }
                    "--lr" => {
                        if i + 1 < args.len() {
                            match args[i + 1].parse::<f32>() {
                                Ok(value) => {
                                    lr = value;
                                    i += 2;
                                }
                                Err(_) => {
                                    println!("Error: Invalid learning rate value");
                                    process::exit(1);
                                }
                            }
                        } else {
                            println!("Error: Missing value for --lr");
                            process::exit(1);
                        }
                    }
                    "--epoch" => {
                        if i + 1 < args.len() {
                            match args[i + 1].parse::<usize>() {
                                Ok(value) => {
                                    epoch = value;
                                    i += 2;
                                }
                                Err(_) => {
                                    println!("Error: Invalid epoch value");
                                    process::exit(1);
                                }
                            }
                        } else {
                            println!("Error: Missing value for --epoch");
                            process::exit(1);
                        }
                    }
                    "--batch-size" => {
                        if i + 1 < args.len() {
                            match args[i + 1].parse::<usize>() {
                                Ok(value) => {
                                    batch_size = value;
                                    i += 2;
                                }
                                Err(_) => {
                                    println!("Error: Invalid batch size value");
                                    process::exit(1);
                                }
                            }
                        } else {
                            println!("Error: Missing value for --batch-size");
                            process::exit(1);
                        }
                    }
                    _ => {
                        println!("Unknown argument: {}", args[i]);
                        print_usage();
                        process::exit(1);
                    }
                }
            }

            if train_data_path.is_empty() || label_path.is_empty() {
                println!("Error: Missing required arguments");
                print_usage();
                process::exit(1);
            }

            if let Err(e) = train(
                train_data_path,
                label_path,
                output_path,
                lr,
                epoch,
                batch_size,
            ) {
                println!("Error during training: {}", e);
                process::exit(1);
            }
        }
        "evaluate" => {
            let mut test_data_path = "";
            let mut label_path = "";
            let mut model_path = "model.bin";

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--test-data" => {
                        if i + 1 < args.len() {
                            test_data_path = &args[i + 1];
                            i += 2;
                        } else {
                            println!("Error: Missing value for --test-data");
                            process::exit(1);
                        }
                    }
                    "--label" => {
                        if i + 1 < args.len() {
                            label_path = &args[i + 1];
                            i += 2;
                        } else {
                            println!("Error: Missing value for --label");
                            process::exit(1);
                        }
                    }
                    "--model" => {
                        if i + 1 < args.len() {
                            model_path = &args[i + 1];
                            i += 2;
                        } else {
                            println!("Error: Missing value for --model");
                            process::exit(1);
                        }
                    }
                    _ => {
                        println!("Unknown argument: {}", args[i]);
                        print_usage();
                        process::exit(1);
                    }
                }
            }

            if test_data_path.is_empty() || label_path.is_empty() {
                println!("Error: Missing required arguments");
                print_usage();
                process::exit(1);
            }

            if let Err(e) = evaluate(test_data_path, label_path, model_path) {
                println!("Error during evaluation: {}", e);
                process::exit(1);
            }
        }
        _ => {
            println!("Unknown command: {}", args[1]);
            print_usage();
            process::exit(1);
        }
    }
}
