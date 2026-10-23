//! macOS window around the ONNX graph page.
//!
//! The page is the same HTML the CLI writes. This binary hosts it in a
//! WKWebView, with a native open panel and file drop.

use ai_model_viewer::graph_page;
use std::path::{Path, PathBuf};
use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::window::{Window, WindowBuilder};
use wry::http::Request;
use wry::{DragDropEvent, WebView, WebViewBuilder};

enum UserEvent {
    OpenDialog,
    ToggleWeights,
    Load(PathBuf),
}

fn main() {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let window = WindowBuilder::new()
        .with_title("AI Model Viewer")
        .with_inner_size(LogicalSize::new(1200.0, 800.0))
        .with_min_inner_size(LogicalSize::new(720.0, 480.0))
        .build(&event_loop)
        .unwrap_or_else(|err| {
            eprintln!("ai-model-viewer-mac: 无法创建窗口: {err}");
            std::process::exit(1);
        });

    let proxy = event_loop.create_proxy();
    let open_proxy = proxy.clone();
    let weights_proxy = proxy.clone();
    let drop_proxy = proxy.clone();

    let webview = WebViewBuilder::new()
        .with_html(welcome_page())
        .with_ipc_handler(
            move |request: Request<String>| match request.body().as_str() {
                "open" => {
                    let _ = open_proxy.send_event(UserEvent::OpenDialog);
                }
                "weights" => {
                    let _ = weights_proxy.send_event(UserEvent::ToggleWeights);
                }
                _ => {}
            },
        )
        .with_drag_drop_handler(move |event| {
            if let DragDropEvent::Drop { paths, .. } = event {
                if let Some(path) = paths.into_iter().next() {
                    let _ = drop_proxy.send_event(UserEvent::Load(path));
                }
            }
            true
        })
        .build(&window)
        .unwrap_or_else(|err| {
            eprintln!("ai-model-viewer-mac: 无法创建 WebView: {err}");
            std::process::exit(1);
        });

    let mut weights = false;
    let mut current = std::env::args().nth(1).map(PathBuf::from);
    if let Some(path) = current.clone() {
        if path.is_file() {
            if let Err(err) = present(&webview, &window, &path, weights) {
                let _ = webview.load_html(&message_page("无法打开模型", &err));
                current = None;
            }
        } else {
            current = None;
        }
    }

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::Opened { urls } => {
                if let Some(path) = urls.into_iter().find_map(|url| url.to_file_path().ok()) {
                    load(&webview, &window, path, weights, &mut current);
                }
            }
            Event::UserEvent(UserEvent::OpenDialog) => {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("打开模型")
                    .add_filter(
                        "AI 模型",
                        &[
                            "onnx",
                            "safetensors",
                            "gguf",
                            "tflite",
                            "pt",
                            "pth",
                            "bin",
                            "pb",
                            "keras",
                            "h5",
                            "hdf5",
                            "json",
                        ],
                    )
                    .pick_file()
                {
                    load(&webview, &window, path, weights, &mut current);
                }
            }
            Event::UserEvent(UserEvent::ToggleWeights) => {
                if let Some(path) = current.clone() {
                    weights = !weights;
                    load(&webview, &window, path, weights, &mut current);
                }
            }
            Event::UserEvent(UserEvent::Load(path)) => {
                load(&webview, &window, path, weights, &mut current);
            }
            Event::WindowEvent {
                event: WindowEvent::DroppedFile(path),
                ..
            } => load(&webview, &window, path, weights, &mut current),
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn load(
    webview: &WebView,
    window: &Window,
    path: PathBuf,
    weights: bool,
    current: &mut Option<PathBuf>,
) {
    match present(webview, window, &path, weights) {
        Ok(()) => *current = Some(path),
        Err(err) => {
            let _ = webview.load_html(&message_page("无法打开模型", &err));
        }
    }
}

fn present(webview: &WebView, window: &Window, path: &Path, weights: bool) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|err| err.to_string())?;
    let html = graph_page(&bytes, weights).map_err(|err| err.to_string())?;
    webview.load_html(&html).map_err(|err| err.to_string())?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("Model");
    window.set_title(&format!("{name} — AI Model Viewer"));
    Ok(())
}

fn welcome_page() -> String {
    message_page(
        "AI Model Viewer",
        "打开模型文件（ONNX、Safetensors、GGUF、TFLite、PyTorch、TensorFlow、Keras），或把文件拖进窗口。",
    )
}

fn message_page(title: &str, body: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #141414; color: #ececec;
    font: 15px/1.5 ui-sans-serif, system-ui, sans-serif; }}
  main {{ min-height: 100%; display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 12px; }}
  button {{ background: #1d1d1d; color: inherit; border: 1px solid #3a3a3a;
    border-radius: 4px; padding: 8px 14px; font: inherit; }}
  p {{ margin: 0; color: #bdbdbd; max-width: 420px; text-align: center; }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p>{body}</p>
  <button type="button" onclick="window.ipc &amp;&amp; window.ipc.postMessage('open')">打开…</button>
</main>
</body>
</html>"#,
        title = escape(title),
        body = escape(body),
    )
}

fn escape(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            ch => out.push(ch),
        }
    }
    out
}
