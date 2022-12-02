use csv::Reader;
use std::fs::File;
use ndarray::{ Array, Array1, Array2 };

use linfa::Dataset;
use linfa_trees::DecisionTree;
use linfa::prelude::*;
use linfa_svm::Svm ;
use linfa_logistic::LogisticRegression;


fn main() {
    let train_data_path = "./titanic_dataset/trainClean.csv";
    let dataset = get_dataset(train_data_path);
    
    let (train, test) = dataset.split_with_ratio(0.8);
    //println!("train: {:?} \n test: {:?} \n",train, test);

    let model = DecisionTree::params()
        .fit(&train).unwrap();
        println!("Decision Tree");
    let accuracy = model.predict(&test).confusion_matrix(&test).unwrap().accuracy();
    println!("Decision Tree accuracy: {}",accuracy);

    let logstic_model = LogisticRegression::default()
        .max_iterations(150)
        .fit(&train)
        .unwrap();
    println!("Logistic Regression");

    let logstic_accuracy = logstic_model.predict(&test).confusion_matrix(&test).unwrap().accuracy();
    println!("Logistic Regression accuracy: {}",logstic_accuracy);

    let svm_model = Svm::<_, bool>::params()
        .linear_kernel()
        .fit(&train).unwrap();
    println!("SVM");
    let svm_accuracy = svm_model.predict(&test).confusion_matrix(&test).unwrap().accuracy();
    println!("SVM accuracy: {}",svm_accuracy);

    
}

fn get_dataset(data_path: &str) -> Dataset<f32, bool, ndarray::Dim<[usize; 1]>> {
    let mut reader = Reader::from_path(data_path).unwrap();

    let headers = get_header(&mut reader);
    let data = get_data(&mut reader);
    let target_index = headers.len() - 1;
    println!("{:?}",headers);
    let features = headers[0..target_index].to_vec();
    
    let records = get_records(&data, target_index,(data.len(),target_index));

    let targets = get_targets(&data, target_index);
    println!("dataset shape: ({},{})\n",data.len(),headers.len());
     
    

    return Dataset::new(records, targets)
        .with_feature_names(features); 
}

fn get_header(reader: &mut Reader<File>) -> Vec<String>{
    return reader.headers()
            .unwrap()
            .iter()
            .map(|r| r.to_owned())
            .collect();
}

fn get_data(reader: &mut Reader<File>) -> Vec<Vec<f32>> {
    return reader
            .records()
            .map(|r| 
                r.unwrap().iter()
                    .map(|field| field.parse::<f32>().unwrap())
                    .collect::<Vec<f32>>()
                ).collect::<Vec<Vec<f32>>>();
}

fn get_records(data: &Vec<Vec<f32>>, target_index: usize, data_shape:(usize,usize)) -> Array2<f32> {
    let mut records: Vec<f32> = vec![];
    for record in data.iter() {
      records.extend_from_slice( &record[0..target_index] );
    }
    return Array::from( records ).into_shape(data_shape).unwrap();

   }
   
   fn get_targets(data: &Vec<Vec<f32>>, target_index: usize) -> Array1<bool> {
    let targets = data
      .iter()
      .map(|record| {
        if record[target_index] == 0.0{
            false
        }else {
            true
        }
    })
      .collect::<Vec<bool>>();
     return Array::from( targets );
   }