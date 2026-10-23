//! Self-contained HTML graph page. Pan, zoom, search, and a detail panel
//! are all in this file so the viewer has no network dependency.

use crate::layout::{estimate_text_px, Kind, Scene, SceneNode};
use crate::onnx::Model;

pub fn render_html(model: &Model, scene: &Scene) -> String {
    let title = if model.graph.name.is_empty() {
        "Model Graph".to_string()
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
    let op_badges = op_counts
        .iter()
        .take(6)
        .map(|(name, count)| {
            format!(
                r#"<span class="op-chip">{} <b>×{}</b></span>"#,
                escape(&clip_to_px(name, 140.0, 11.0)),
                count
            )
        })
        .collect::<Vec<_>>()
        .join("");

    let mut body = String::new();
    body.push_str(&defs());
    for (from, to) in &scene.edges {
        let a = &scene.nodes[*from];
        let b = &scene.nodes[*to];
        let x1 = a.x + a.w / 2.0;
        let y1 = a.y + a.h;
        let x2 = b.x + b.w / 2.0;
        let y2 = b.y;
        let bend = ((y2 - y1).abs() / 2.2).clamp(22.0, 72.0);
        body.push_str(&format!(
            r#"<path class="edge" marker-end="url(#arrow)" d="M {x1:.1} {y1:.1} C {x1:.1} {cy1:.1}, {x2:.1} {cy2:.1}, {x2:.1} {y2:.1}"/>"#,
            cy1 = y1 + bend,
            cy2 = y2 - bend,
        ));
    }
    for (ordinal, node) in scene.nodes.iter().enumerate() {
        body.push_str(&node_markup(ordinal, node));
    }

    let producer = if model.producer_name.is_empty() {
        String::new()
    } else if model.producer_version.is_empty() {
        escape(&model.producer_name)
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
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{
    margin: 0;
    height: 100%;
    overflow: hidden;
    background: #111316;
    color: #e6e8ec;
    font: 13px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", ui-sans-serif, system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
    user-select: none;
  }}
  body {{
    display: flex;
    flex-direction: column;
  }}
  header {{
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    padding: 10px 16px;
    background: rgba(24, 26, 31, 0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid #262930;
    min-height: 54px;
  }}
  .header-meta {{
    min-width: 0;
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .title-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }}
  header h1 {{
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #f5f7fa;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 340px;
  }}
  .meta-line {{
    margin: 0;
    color: #8f96a3;
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .chip-row {{
    display: flex;
    align-items: center;
    gap: 6px;
    overflow: hidden;
    white-space: nowrap;
  }}
  .op-chip {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 1px 7px;
    border-radius: 999px;
    background: #1e222a;
    border: 1px solid #2d323e;
    color: #b7becb;
    font-size: 11px;
    flex-shrink: 0;
  }}
  .op-chip b {{
    color: #79a6f6;
    font-weight: 600;
  }}
  .toolbar {{
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .app-btn {{
    flex-shrink: 0;
    white-space: nowrap;
    background: #20242c;
    color: #e6e8ec;
    border: 1px solid #343946;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.12s, border-color 0.12s;
  }}
  .app-btn:hover {{
    background: #2a2f3a;
    border-color: #485063;
  }}
  #search {{
    flex-shrink: 0;
    background: #171a20;
    color: #e6e8ec;
    border: 1px solid #323744;
    border-radius: 6px;
    padding: 6px 10px;
    width: 210px;
    font-size: 12px;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
  }}
  #search:focus {{
    border-color: #4f8cff;
    box-shadow: 0 0 0 2px rgba(79, 140, 255, 0.2);
  }}
  #main {{
    flex: 1 1 auto;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 310px;
  }}
  #viewport {{
    position: relative;
    overflow: hidden;
    background:
      radial-gradient(circle at 1px 1px, #22262e 1px, transparent 0) 0 0 / 22px 22px,
      #101215;
    cursor: grab;
  }}
  #viewport.dragging {{ cursor: grabbing; }}
  .edge {{
    fill: none;
    stroke: #6e7787;
    stroke-width: 1.4;
    transition: stroke 0.15s;
  }}
  .node {{ cursor: pointer; }}
  .node .card-outline {{
    fill: none;
    stroke: #2e3440;
    stroke-width: 1.2;
    transition: stroke 0.12s, stroke-width 0.12s;
  }}
  .node:hover .card-outline {{
    stroke: #7a869a;
    stroke-width: 1.6;
  }}
  .node.selected .card-outline {{
    stroke: #6aa3ff;
    stroke-width: 2.2;
  }}
  .node.dim {{ opacity: 0.15; }}
  .node text {{
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", ui-sans-serif, system-ui, sans-serif;
    pointer-events: none;
  }}
  aside {{
    border-left: 1px solid #262930;
    background: #16181d;
    padding: 16px;
    overflow-y: auto;
    overflow-x: hidden;
    user-select: text;
  }}
  aside h2 {{
    margin: 0 0 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid #262930;
    font-size: 14px;
    font-weight: 600;
    color: #f3f5f8;
    overflow-wrap: anywhere;
  }}
  #panel-body {{
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  #panel-body li {{
    padding: 8px 10px;
    background: #1d2027;
    border: 1px solid #282c36;
    border-radius: 6px;
    font-size: 12px;
    line-height: 1.45;
    color: #d7dce5;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  #panel-body li .kv-tag {{
    display: inline-block;
    margin-right: 6px;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    background: #283040;
    color: #8ab4f8;
  }}
  #tools {{
    position: absolute;
    left: 14px;
    bottom: 14px;
    display: flex;
    gap: 6px;
    background: rgba(24, 26, 31, 0.88);
    backdrop-filter: blur(8px);
    padding: 4px;
    border-radius: 8px;
    border: 1px solid #2d323e;
  }}
  #tools button {{
    background: transparent;
    color: #dcdfe6;
    border: none;
    border-radius: 5px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
  }}
  #tools button:hover {{
    background: #2a2f3a;
  }}
</style>
</head>
<body>
<header>
  <div class="header-meta">
    <div class="title-row">
      <h1 title="{title}">{title}</h1>
      <p class="meta-line">ir {ir} · {nodes} ops · {inits} initializers{functions}{producer}{opsets}</p>
    </div>
    <div class="chip-row">{op_badges}</div>
  </div>
  <div class="toolbar" id="toolbar">
    <input id="search" type="search" placeholder="过滤算子、名字、属性" autofocus>
  </div>
</header>
<div id="main">
  <div id="viewport">
    <svg id="surface" width="100%" height="100%" data-width="{width}" data-height="{height}">
      <g id="scene">{body}</g>
    </svg>
    <div id="tools">
      <button type="button" id="zoom-in" title="放大">+</button>
      <button type="button" id="zoom-out" title="缩小">−</button>
      <button type="button" id="fit" title="适配窗口">适配</button>
    </div>
  </div>
  <aside>
    <h2 id="panel-title">模型概览</h2>
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
  if (rect.width <= 0 || rect.height <= 0) return;
  const sx = (rect.width - 56) / Math.max(box.width, 1);
  const sy = (rect.height - 56) / Math.max(box.height, 1);
  scale = Math.max(0.15, Math.min(1.35, Math.min(sx, sy)));
  x = (rect.width - box.width * scale) / 2 - box.x * scale;
  y = Math.max(24, (rect.height - box.height * scale) / 2) - box.y * scale;
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
    const sep = line.indexOf(": ");
    if (sep > 0 && sep < 14) {{
      const tag = document.createElement("span");
      tag.className = "kv-tag";
      tag.textContent = line.slice(0, sep);
      item.appendChild(tag);
      item.appendChild(document.createTextNode(line.slice(sep + 2)));
    }} else {{
      item.textContent = line;
    }}
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
window.addEventListener("resize", fit);
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
  const toolbar = document.getElementById("toolbar");
  const search = document.getElementById("search");
  function appButton(label, message) {{
    const button = document.createElement("button");
    button.type = "button";
    button.className = "app-btn";
    button.textContent = label;
    button.onclick = () => window.ipc.postMessage(message);
    toolbar.insertBefore(button, search);
  }}
  appButton("打开…", "open");
  appButton("切换权重", "weights");
}}
apply();
requestAnimationFrame(fit);
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
        op_badges = op_badges,
        hint = "点击画布中的算子或张量节点查看输入、输出、权重和属性详情。使用鼠标滚轮或触控板缩放，按住画布拖动。",
        body = body,
        width = scene.width,
        height = scene.height,
        json = json_records(&scene.nodes),
    )
}

fn defs() -> String {
    r##"<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#6e7787"/></marker></defs>"##.into()
}

fn node_markup(ordinal: usize, node: &SceneNode) -> String {
    let header_fill = match node.kind {
        Kind::Input => "#24527a",
        Kind::Output => "#2b684c",
        Kind::Initializer => "#444952",
        Kind::Unresolved => "#6b3030",
        Kind::Operator => op_color(&node.op_type),
    };
    let body_fill = match node.kind {
        Kind::Input => "#182430",
        Kind::Output => "#182921",
        Kind::Initializer => "#1e2024",
        Kind::Unresolved => "#291a1a",
        Kind::Operator => "#1c1f26",
    };

    let clip_id = format!("nc-{ordinal}");
    let avail_px = (node.w - 22.0).max(80.0);
    let header_h = if node.lines.is_empty() {
        node.h
    } else {
        26.0
    };

    let mut inner = String::new();
    // Base card background
    inner.push_str(&format!(
        r#"<rect x="0" y="0" width="{w:.1}" height="{h:.1}" rx="6" fill="{body_fill}"/>"#,
        w = node.w,
        h = node.h,
    ));
    // Header band
    inner.push_str(&format!(
        r#"<rect x="0" y="0" width="{w:.1}" height="{hh:.1}" fill="{header_fill}"/>"#,
        w = node.w,
        hh = header_h,
    ));
    // Title text
    inner.push_str(&format!(
        r##"<text x="11" y="17.5" font-size="12.5" font-weight="600" fill="#f7f9fc">{title}</text>"##,
        title = escape(&clip_to_px(&node.title, avail_px, 12.5)),
    ));
    // Body lines
    for (index, line) in node.lines.iter().enumerate() {
        inner.push_str(&format!(
            r##"<text x="11" y="{y:.1}" font-size="11" fill="#c7cdd8">{line}</text>"##,
            y = 43.0 + index as f64 * 17.0,
            line = escape(&clip_to_px(line, avail_px, 11.0)),
        ));
    }

    format!(
        r#"<g class="node" data-id="{id}"><defs><clipPath id="{clip_id}"><rect x="0" y="0" width="{w:.1}" height="{h:.1}" rx="6"/></clipPath></defs><svg x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" overflow="hidden"><g clip-path="url(#{clip_id})">{inner}</g></svg><rect class="card-outline" x="{x:.1}" y="{y:.1}" width="{w:.1}" height="{h:.1}" rx="6"/></g>"#,
        id = escape(&node.id),
        clip_id = clip_id,
        x = node.x,
        y = node.y,
        w = node.w,
        h = node.h,
        inner = inner,
    )
}

fn op_color(op: &str) -> &'static str {
    const COLORS: [&str; 6] = [
        "#36557c", "#544575", "#356654", "#684b36", "#733f4b", "#35565c",
    ];
    let mut hash = 0usize;
    for byte in op.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as usize);
    }
    COLORS[hash % COLORS.len()]
}

/// Truncate a string with `…` so its rendered width stays within `max_px`.
fn clip_to_px(text: &str, max_px: f64, font_px: f64) -> String {
    if estimate_text_px(text, font_px) <= max_px {
        return text.to_string();
    }
    let ellipsis_px = font_px * 0.9;
    let budget = (max_px - ellipsis_px).max(font_px);
    let mut kept = String::new();
    let mut used = 0.0f64;
    for ch in text.chars() {
        let ch_buf = [ch];
        let s: String = ch_buf.iter().collect();
        let w = estimate_text_px(&s, font_px);
        if used + w > budget {
            break;
        }
        kept.push(ch);
        used += w;
    }
    format!("{kept}…")
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
        assert!(html.contains("clipPath id=\"nc-"));
    }

    #[test]
    fn clips_long_cjk_and_uppercase_to_fit_node_box() {
        let long_cjk = "这是一个非常非常长的中文属性名称或张量描述信息不会溢出节点边框";
        let clipped = clip_to_px(long_cjk, 180.0, 11.0);
        assert!(clipped.ends_with('…'));
        assert!(estimate_text_px(&clipped, 11.0) <= 185.0);
    }
}
