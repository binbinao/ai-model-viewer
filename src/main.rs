fn main() {
    if let Err(err) = ai_model_viewer::run(std::env::args().skip(1)) {
        eprintln!("ai-model-viewer: {err}");
        std::process::exit(1);
    }
}
