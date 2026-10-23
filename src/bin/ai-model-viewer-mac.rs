//! macOS window around the model graph page.
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
        .with_inner_size(LogicalSize::new(1240.0, 820.0))
        .with_min_inner_size(LogicalSize::new(760.0, 500.0))
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
        "选择或拖入模型文件以可视化计算图与权重层级。",
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
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{
    margin: 0;
    height: 100%;
    overflow: hidden;
    background:
      radial-gradient(circle at 1px 1px, #22262e 1px, transparent 0) 0 0 / 22px 22px,
      #101215;
    color: #e6e8ec;
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", ui-sans-serif, system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
    user-select: none;
  }}
  main {{
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }}
  .card {{
    width: 100%;
    max-width: 480px;
    padding: 36px 32px;
    border-radius: 14px;
    background: rgba(24, 27, 34, 0.92);
    border: 1px dashed #363d4d;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 14px;
    text-align: center;
  }}
  h1 {{
    margin: 0;
    font-size: 20px;
    font-weight: 600;
    color: #f5f7fa;
  }}
  p {{
    margin: 0;
    color: #9ba3b2;
    max-width: 380px;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  .formats {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 6px;
    margin: 4px 0 8px;
  }}
  .badge {{
    padding: 2px 8px;
    border-radius: 999px;
    background: #1f242f;
    border: 1px solid #2f3646;
    color: #8ab4f8;
    font-size: 11px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  button {{
    background: #2b63d9;
    color: #ffffff;
    border: 1px solid #3d75eb;
    border-radius: 7px;
    padding: 8px 20px;
    font: 500 13px/1.4 inherit;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.12s;
  }}
  button:hover {{
    background: #366fe6;
  }}
</style>
</head>
<body>
<main>
  <div class="card">
    <h1>{title}</h1>
    <p>{body}</p>
    <div class="formats">
      <span class="badge">.onnx</span>
      <span class="badge">.safetensors</span>
      <span class="badge">.gguf</span>
      <span class="badge">.tflite</span>
      <span class="badge">.pt / .pth</span>
      <span class="badge">.pb</span>
      <span class="badge">.keras / .h5</span>
    </div>
    <button type="button" onclick="window.ipc &amp;&amp; window.ipc.postMessage('open')">打开模型文件…</button>
  </div>
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
