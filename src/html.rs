//! Self-contained HTML graph page. Pan, zoom, search, and a detail panel
//! are all in this file so the viewer has no network dependency.

use crate::layout::{Kind, Scene, SceneNode};
use crate::onnx::Model;

pub fn render_html(model: &Model, scene: &Scene) -> String {
    let title = if model.graph.name.is_empty() {
        "ONNX graph".to_string()
    } else {
        model.graph.name.clone()
    };
    let mut op_counts: Vec<(String, usize)> = Vec::new();
    for node in &model.graph.nodes {
        let name = if node.op_type.is_empty() {
            "(empty)".to_string()
        } else {
            node.op_type.clone()
        };
        if let Some((_, count)) = op_counts.iter_mut().find(|(op, _)| *op == name) {
            *count += 1;
        } else {
            op_counts.push((name, 1));
        }
    }
    op_counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let op_summary = op_counts
        .iter()
        .take(8)
        .map(|(name, count)| format!("{name} × {count}"))
        .collect::<Vec<_>>()
        .join("  ");

    let mut body = String::new();
    body.push_str(&defs());
    for (from, to) in &scene.edges {
        let a = &scene.nodes[*from];
        let b = &scene.nodes[*to];
        let x1 = a.x + a.w / 2.0;
        let y1 = a.y + a.h;
        let x2 = b.x + b.w / 2.0;
        let y2 = b.y;
        let bend = ((y2 - y1) / 2.0).max(24.0);
        body.push_str(&format!(
            r#"<path class="edge" d="M {x1:.1} {y1:.1} C {x1:.1} {cy1:.1}, {x2:.1} {cy2:.1}, {x2:.1} {y2:.1}"/>"#,
            cy1 = y1 + bend,
            cy2 = y2 - bend,
        ));
    }
    for node in &scene.nodes {
        body.push_str(&node_markup(node));
    }

    let producer = if model.producer_name.is_empty() {
        String::new()
    } else {
        format!(
            "{} {}",
            escape(&model.producer_name),
            escape(&model.producer_version)
        )
    };
    let opsets = model
        .opsets
        .iter()
        .map(|opset| {
            let domain = if opset.domain.is_empty() {
                "ai.onnx".to_string()
            } else {
                opset.domain.clone()
            };
            format!("{}:{}", escape(&domain), opset.version)
        })
        .collect::<Vec<_>>()
        .join("  ");
    let functions = if model.functions.is_empty() {
        String::new()
    } else {
        format!(" · functions {}", model.functions.len())
    };

    format!(
        r##"<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  html, body {{ margin: 0; height: 100%; background: #141414; color: #ececec; font: 13px/1.4 ui-sans-serif, system-ui, sans-serif; }}
  header {{ display: flex; gap: 16px; align-items: center; padding: 10px 14px; border-bottom: 1px solid #2a2a2a; }}
  header h1 {{ margin: 0; font-size: 14px; font-weight: 600; }}
  header p {{ margin: 0; color: #9a9a9a; }}
  #search {{ margin-left: auto; background: #1d1d1d; color: inherit; border: 1px solid #3a3a3a; border-radius: 4px; padding: 6px 8px; width: 220px; }}
  .app-btn {{ background: #1d1d1d; color: inherit; border: 1px solid #3a3a3a; border-radius: 4px; padding: 6px 10px; }}
  #main {{ display: grid; grid-template-columns: 1fr 280px; height: calc(100% - 58px); }}
  #viewport {{ position: relative; overflow: hidden; background: #101010; cursor: grab; }}
  #viewport.dragging {{ cursor: grabbing; }}
  .edge {{ fill: none; stroke: #6a6a6a; stroke-width: 1.25; }}
  .node {{ cursor: pointer; }}
  .node rect {{ stroke: #0000; stroke-width: 2; }}
  .node.selected rect {{ stroke: #f2f2f2; }}
  .node.dim {{ opacity: 0.15; }}
  .node text {{ fill: #f5f5f5; font-family: ui-sans-serif, system-ui, sans-serif; }}
  aside {{ border-left: 1px solid #2a2a2a; padding: 14px; overflow: auto; }}
  aside h2 {{ margin: 0 0 8px; font-size: 14px; }}
  aside ul {{ margin: 0; padding-left: 16px; }}
  aside li {{ margin: 3px 0; word-break: break-word; }}
  #tools {{ position: absolute; left: 12px; bottom: 12px; display: flex; gap: 6px; }}
  #tools button {{ background: #1d1d1d; color: inherit; border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px 8px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>{title}</h1>
    <p>ir {ir} · {nodes} ops · {inits} initializers{functions}{producer}{opsets}</p>
    <p>{op_summary}</p>
  </div>
  <input id="search" type="search" placeholder="过滤算子、名字、属性" autofocus>
</header>
<div id="main">
  <div id="viewport">
    <svg id="surface" width="100%" height="100%" data-width="{width}" data-height="{height}">
      <g id="scene">{body}</g>
    </svg>
    <div id="tools">
      <button type="button" id="zoom-in">+</button>
      <button type="button" id="zoom-out">−</button>
      <button type="button" id="fit">适配</button>
    </div>
  </div>
  <aside>
    <h2 id="panel-title">图</h2>
    <ul id="panel-body"><li>{hint}</li></ul>
  </aside>
</div>
<script id="model" type="application/json">{json}</script>
<script>
const records = JSON.parse(document.getElementById("model").textContent);
const byId = Object.fromEntries(records.map((node) => [node.id, node]));
const viewport = document.getElementById("viewport");
const scene = document.getElementById("scene");
const panelTitle = document.getElementById("panel-title");
const panelBody = document.getElementById("panel-body");
let x = 24, y = 24, scale = 1, drag = null;

function apply() {{
  scene.setAttribute("transform", `translate(${{x}} ${{y}}) scale(${{scale}})`);
}}
function fit() {{
  const box = scene.getBBox();
  const rect = viewport.getBoundingClientRect();
  const sx = (rect.width - 48) / Math.max(box.width, 1);
  const sy = (rect.height - 48) / Math.max(box.height, 1);
  scale = Math.max(0.15, Math.min(1.4, Math.min(sx, sy)));
  x = (rect.width - box.width * scale) / 2 - box.x * scale;
  y = 24 - box.y * scale;
  apply();
}}
function show(id) {{
  for (const el of scene.querySelectorAll(".node")) el.classList.toggle("selected", el.dataset.id === id);
  const node = byId[id];
  if (!node) return;
  panelTitle.textContent = node.title;
  panelBody.replaceChildren();
  const lines = node.detail.length ? node.detail : ["没有更多字段"];
  for (const line of lines) {{
    const item = document.createElement("li");
    item.textContent = line;
    panelBody.appendChild(item);
  }}
}}
viewport.addEventListener("pointerdown", (event) => {{
  const node = event.target.closest(".node");
  if (node) {{
    show(node.dataset.id);
    return;
  }}
  drag = {{ px: event.clientX, py: event.clientY, x, y }};
  viewport.classList.add("dragging");
  viewport.setPointerCapture(event.pointerId);
}});
viewport.addEventListener("pointermove", (event) => {{
  if (!drag) return;
  x = drag.x + event.clientX - drag.px;
  y = drag.y + event.clientY - drag.py;
  apply();
}});
function endDrag() {{
  drag = null;
  viewport.classList.remove("dragging");
}}
viewport.addEventListener("pointerup", endDrag);
viewport.addEventListener("pointercancel", endDrag);
viewport.addEventListener("wheel", (event) => {{
  event.preventDefault();
  const next = Math.min(2.8, Math.max(0.15, scale * (event.deltaY < 0 ? 1.1 : 0.9)));
  const rect = viewport.getBoundingClientRect();
  const cx = event.clientX - rect.left;
  const cy = event.clientY - rect.top;
  x = cx - (cx - x) * (next / scale);
  y = cy - (cy - y) * (next / scale);
  scale = next;
  apply();
}}, {{ passive: false }});
document.getElementById("zoom-in").onclick = () => {{ scale = Math.min(2.8, scale * 1.15); apply(); }};
document.getElementById("zoom-out").onclick = () => {{ scale = Math.max(0.15, scale / 1.15); apply(); }};
document.getElementById("fit").onclick = fit;
document.getElementById("search").addEventListener("input", (event) => {{
  const query = event.target.value.trim().toLowerCase();
  for (const el of scene.querySelectorAll(".node")) {{
    const node = byId[el.dataset.id];
    const blob = [node.title, node.op, ...node.lines, ...node.detail].join(" ").toLowerCase();
    el.classList.toggle("dim", query.length > 0 && !blob.includes(query));
  }}
}});
if (window.ipc) {{
  const header = document.querySelector("header");
  const search = document.getElementById("search");
  function appButton(label, message) {{
    const button = document.createElement("button");
    button.type = "button";
    button.className = "app-btn";
    button.textContent = label;
    button.onclick = () => window.ipc.postMessage(message);
    header.insertBefore(button, search);
  }}
  appButton("打开…", "open");
  appButton("权重", "weights");
}}
apply();
fit();
</script>
</body>
</html>
"##,
        title = escape(&title),
        ir = model.ir_version,
        nodes = model.graph.nodes.len(),
        inits = model.graph.initializers.len(),
        functions = functions,
        producer = if producer.is_empty() {
            String::new()
        } else {
            format!(" · {producer}")
        },
        opsets = if opsets.is_empty() {
            String::new()
        } else {
            format!(" · {opsets}")
        },
        op_summary = escape(&op_summary),
        hint = "点击节点查看输入、输出和属性。滚轮缩放，拖动画布。",
        body = body,
        width = scene.width,
        height = scene.height,
        json = json_records(&scene.nodes),
    )
}

fn defs() -> String {
    r##"<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 8 5 L 0 9 z" fill="#6a6a6a"/></marker></defs>"##.into()
}

fn node_markup(node: &SceneNode) -> String {
    let fill = match node.kind {
        Kind::Input => "#24527a",
        Kind::Output => "#2f6b4f",
        Kind::Initializer => "#4a4e55",
        Kind::Unresolved => "#6b3030",
        Kind::Operator => op_color(&node.op_type),
    };
    let mut text = format!(
        r##"<text x="{x}" y="{y}" font-size="13" font-weight="600">{title}</text>"##,
        x = node.x + 10.0,
        y = node.y + 18.0,
        title = escape(&clip(&node.title, 24)),
    );
    for (index, line) in node.lines.iter().enumerate() {
        text.push_str(&format!(
            r##"<text x="{x}" y="{y}" font-size="11" fill="#d5d5d5">{line}</text>"##,
            x = node.x + 10.0,
            y = node.y + 36.0 + index as f64 * 16.0,
            line = escape(&clip(line, 28)),
        ));
    }
    format!(
        r#"<g class="node" data-id="{id}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}"/>{text}</g>"#,
        id = escape(&node.id),
        x = node.x,
        y = node.y,
        w = node.w,
        h = node.h,
        fill = fill,
        text = text,
    )
}

fn op_color(op: &str) -> &'static str {
    const COLORS: [&str; 6] = [
        "#3d5a80", "#5c4d7a", "#3f6f5b", "#6b4f3a", "#7a4450", "#3e5c62",
    ];
    let mut hash = 0usize;
    for byte in op.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
    }
    COLORS[hash % COLORS.len()]
}

fn json_records(nodes: &[SceneNode]) -> String {
    let mut out = String::from("[");
    for (index, node) in nodes.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            r#"{{"id":{id},"title":{title},"op":{op},"lines":{lines},"detail":{detail}}}"#,
            id = json_string(&node.id),
            title = json_string(&node.title),
            op = json_string(&node.op_type),
            lines = json_array(&node.lines),
            detail = json_array(&node.detail),
        ));
    }
    out.push(']');
    out
}

fn json_array(items: &[String]) -> String {
    let mut out = String::from("[");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&json_string(item));
    }
    out.push(']');
    out
}

fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '<' => out.push_str("\\u003c"),
            ch if (ch as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
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

fn clip(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let kept: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::layout;
    use crate::onnx::{parse_model, tiny_model_bytes};

    #[test]
    fn page_names_the_operators() {
        let model = parse_model(&tiny_model_bytes()).unwrap();
        let scene = layout(&model, false);
        let html = render_html(&model, &scene);
        assert!(html.contains("Relu"));
        assert!(html.contains("Add"));
        assert!(html.contains("权重 × 1"));
        assert!(html.contains("ai.onnx:18"));
        assert!(html.contains("axis = 1"));
    }
}
