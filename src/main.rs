use crate::linear_regression::LinearRegression;
use actix_cors::Cors; // Import the CORS middleware
use actix_files::Files;
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

mod linear_regression;

#[derive(Serialize, Deserialize)]
struct PredictionRequest {
    feature1: f64,
    feature2: f64,
    feature3: f64,
}

#[derive(Serialize)]
struct PredictionResponse {
    prediction: f64,
}

async fn predict_handler(
    data: web::Json<PredictionRequest>,
    model: web::Data<Arc<RwLock<LinearRegression>>>,
) -> impl Responder {
    let model = model.read().await;
    let inputs = vec![data.feature1, data.feature2, data.feature3];
    let prediction = model.predict(&inputs);
    HttpResponse::Ok().json(PredictionResponse { prediction })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Server is running at http://127.0.0.1:8080");
    println!("Open your browser and navigate to http://127.0.0.1:8080/index.html");

    // Initialize the model with example coefficients
    let model = LinearRegression {
        coefficients: vec![1.0, 2.0, 3.0],
    };

    HttpServer::new(move || {
        App::new()
            .wrap(
                Cors::default()
                    .allow_any_origin() // Allow requests from any origin
                    .allow_any_method() // Allow any HTTP method
                    .allow_any_header() // Allow any header
                    .max_age(3600), // Cache the preflight response for 1 hour
            )
            .app_data(web::Data::new(Arc::new(RwLock::new(model.clone()))))
            .route("/predict", web::post().to(predict_handler))
            .service(Files::new("/", "./static").index_file("index.html"))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
