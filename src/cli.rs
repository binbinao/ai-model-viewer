use crate::html::render_html;
use crate::layout::layout;
use crate::onnx::{parse_model, OnnxError};
use std::env;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
pub enum CliError {
    Usage(String),
    Onnx(OnnxError),
    Io(io::Error),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message) | Self::Onnx(OnnxError::Message(message)) => {
                write!(f, "{message}")
            }
            Self::Onnx(OnnxError::Proto(err)) => write!(f, "{err}"),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for CliError {}

impl From<OnnxError> for CliError {
    fn from(err: OnnxError) -> Self {
        Self::Onnx(err)
    }
}

impl From<io::Error> for CliError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

struct Options {
    input: PathBuf,
    output: Option<PathBuf>,
    text_only: bool,
    show_weights: bool,
    open: bool,
}

pub fn run<I, S>(args: I) -> Result<(), CliError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let options = parse_args(args)?;
    let bytes = fs::read(&options.input)?;
    let model = parse_model(&bytes)?;

    let wrote = if options.text_only && options.output.is_none() {
        None
    } else {
        let scene = layout(&model, options.show_weights);
        let html = render_html(&model, &scene);
        let output = match &options.output {
            Some(path) => path.clone(),
            None => default_output(&options.input)?,
        };
        fs::write(&output, html)?;
        if options.open {
            open_page(&output)?;
        }
        Some(output)
    };

    print_line(&render_text(&model))?;
    if let Some(output) = wrote {
        print_line(&format!("写入 {}", output.display()))?;
    }
    Ok(())
}

fn print_line(text: &str) -> Result<(), CliError> {
    let mut stdout = io::stdout();
    match writeln!(stdout, "{text}") {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(err) => Err(err.into()),
    }
}

fn parse_args<I, S>(args: I) -> Result<Options, CliError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let args: Vec<String> = args
        .into_iter()
        .map(|arg| arg.as_ref().to_string())
        .collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") || args.is_empty() {
        return Err(CliError::Usage(help()));
    }

    let mut input = None;
    let mut output = None;
    let mut text_only = false;
    let mut show_weights = false;
    let mut open = false;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        match arg.as_str() {
            "--text" => text_only = true,
            "--weights" => show_weights = true,
            "--open" => open = true,
            "-o" | "--output" => {
                index += 1;
                let path = args
                    .get(index)
                    .ok_or_else(|| CliError::Usage("-o 后面需要一个 html 路径".into()))?;
                output = Some(PathBuf::from(path));
            }
            "--" => {
                return Err(CliError::Usage(help()));
            }
            other if other.starts_with('-') => {
                return Err(CliError::Usage(format!("未知参数 {other}\n\n{}", help())));
            }
            other => {
                if input.is_some() {
                    return Err(CliError::Usage("一次只能打开一个模型".into()));
                }
                input = Some(PathBuf::from(other));
            }
        }
        index += 1;
    }

    let input = input.ok_or_else(|| CliError::Usage(help()))?;
    if !input.is_file() {
        return Err(CliError::Usage(format!("找不到文件 {}", input.display())));
    }
    Ok(Options {
        input,
        output,
        text_only,
        show_weights,
        open,
    })
}

fn help() -> String {
    "\
用法: ai-model-viewer <模型文件> [-o 文件.html] [--text] [--weights] [--open]

支持的格式: ONNX (.onnx)、Safetensors (.safetensors)、GGUF (.gguf)、
           TFLite (.tflite)、PyTorch / TorchScript (.pt/.pth/.bin)、
           TensorFlow GraphDef (.pb)、Keras (.keras/.h5/.json)。

打印模型图摘要，并写出一个可平移缩放的 HTML 图。
权重默认不画成独立节点，算子/模块卡片上会标出权重个数。--weights 把权重节点画出来。"
        .into()
}

fn default_output(input: &Path) -> Result<PathBuf, CliError> {
    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    let name = format!("{stem}.html");
    Ok(env::current_dir()?.join(name))
}

pub fn render_text(model: &crate::onnx::Model) -> String {
    let mut lines = Vec::new();
    let name = if model.graph.name.is_empty() {
        "(unnamed)"
    } else {
        model.graph.name.as_str()
    };
    lines.push(format!("graph {name}"));
    lines.push(format!(
        "ir {}  producer {} {}",
        model.ir_version,
        dash(&model.producer_name),
        model.producer_version
    ));
    if !model.opsets.is_empty() {
        let opsets: Vec<String> = model
            .opsets
            .iter()
            .map(|opset| {
                let domain = if opset.domain.is_empty() {
                    "ai.onnx"
                } else {
                    opset.domain.as_str()
                };
                format!("{domain}:{}", opset.version)
            })
            .collect();
        lines.push(format!("opset {}", opsets.join("  ")));
    }
    lines.push(format!(
        "inputs {}   initializers {}   nodes {}   outputs {}",
        model.graph.inputs.len(),
        model.graph.initializers.len(),
        model.graph.nodes.len(),
        model.graph.outputs.len()
    ));
    for input in &model.graph.inputs {
        let role = if model
            .graph
            .initializers
            .iter()
            .any(|tensor| tensor.name == input.name)
        {
            "init"
        } else {
            "in  "
        };
        lines.push(format!("  {role} {} {}", input.name, input.type_text));
    }
    for output in &model.graph.outputs {
        lines.push(format!("  out {} {}", output.name, output.type_text));
    }
    for node in &model.graph.nodes {
        let name = if node.name.is_empty() {
            String::new()
        } else {
            format!(" {}", node.name)
        };
        lines.push(format!("  {}{name}", dash(&node.op_type)));
    }
    if !model.functions.is_empty() {
        lines.push(format!("functions {}", model.functions.len()));
        for function in &model.functions {
            lines.push(format!(
                "  fn {} {} ({} nodes)",
                dash(&function.domain),
                dash(&function.name),
                function.node_count
            ));
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::tiny_model_bytes;

    #[test]
    fn writes_html_summary() {
        let dir = env::temp_dir().join(format!("ai-model-viewer-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("tiny.onnx");
        let output = dir.join("tiny.html");
        fs::write(&input, tiny_model_bytes()).unwrap();
        run([input.to_str().unwrap(), "-o", output.to_str().unwrap()]).unwrap();
        let html = fs::read_to_string(&output).unwrap();
        assert!(html.contains("Relu"));
        assert!(html.contains("data-width="));
    }
}

fn dash(text: &str) -> &str {
    if text.is_empty() {
        "-"
    } else {
        text
    }
}

fn open_page(path: &Path) -> Result<(), CliError> {
    let program = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "cmd"
    } else {
        "xdg-open"
    };
    let mut command = Command::new(program);
    if cfg!(target_os = "windows") {
        command.args(["/C", "start", ""]);
    }
    let status = command.arg(path).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(CliError::Usage(format!(
            "无法打开 {}（{} 退出码 {:?}）",
            path.display(),
            program,
            status.code()
        )))
    }
}
