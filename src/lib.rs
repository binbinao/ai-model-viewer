mod cli;
pub mod gguf;
mod hierarchy;
mod html;
mod json;
pub mod keras;
mod layout;
mod onnx;
mod proto;
pub mod pytorch;
pub mod safetensors;
pub mod tf_pb;
pub mod tflite;
mod zip_read;

pub use cli::run;
pub use onnx::{parse_model, Model, OnnxError};

pub fn graph_page(bytes: &[u8], show_weights: bool) -> Result<String, OnnxError> {
    let model = parse_model(bytes)?;
    let scene = layout::layout(&model, show_weights);
    Ok(html::render_html(&model, &scene))
}
