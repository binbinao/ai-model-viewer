# ai-model-viewer

一个多格式 AI 模型图查看器。读取模型结构或权重元数据，在终端打印摘要，并写出可以平移、缩放、搜索、点选详情的独立 HTML 图。

## 支持的模型格式

| 格式 | 扩展名 | 解析方式 |
|---|---|---|
| **ONNX** | `.onnx` | 读取二进制 `ModelProto`，解析计算图算子、输入输出、权重与属性 |
| **Safetensors** | `.safetensors` | 读取 JSON 头部，按层命名自动聚类为层级节点图并统计张量形状与 dtype |
| **GGUF** | `.gguf` | 读取 GGUF v1–v3 头部、架构元数据（`general.*`）与量化张量目录，生成层级模块图 |
| **TensorFlow Lite** | `.tflite` | 直接解析 FlatBuffers（`TFL3`/`TFL2`）子图、算子码、输入输出与常量缓冲 |
| **PyTorch / TorchScript** | `.pt`, `.pth`, `.bin` | 解析 ZIP 归档中的 TorchScript `code/*.py` 计算图或静态提取 `state_dict` 张量树 |
| **TensorFlow GraphDef / SavedModel** | `.pb` | 读取 `GraphDef` 或 `SavedModel` 中的 `NodeDef` 拓扑与属性 |
| **Keras / HDF5** | `.keras`, `.h5`, `.hdf5`, `.json` | 解析 Keras v3 `.keras` 归档、内嵌 `model_config` 的 `.h5` 或独立拓扑 JSON |

## 用法

```bash
cargo run --bin ai-model-viewer -- path/to/model.onnx
cargo run --bin ai-model-viewer -- path/to/model.safetensors -o graph.html --open
cargo run --bin ai-model-viewer -- path/to/model.gguf --text
cargo run --bin ai-model-viewer -- path/to/model.tflite --weights
```

| 选项 | 作用 |
|---|---|
| `-o, --output` | HTML 输出路径。默认是当前目录下的 `<模型名>.html` |
| `--text` | 只打印摘要 |
| `--weights` | 把 initializer / 权重张量画成独立节点。默认只在算子或层级卡片上标权重个数 |
| `--open` | 写完后用系统浏览器打开 |

macOS 应用包在同一套图页面上加了一个窗口：

```bash
./scripts/bundle-macos.sh
open "dist/AI Model Viewer.app"
open "dist/AI Model Viewer.app" --args path/to/model.onnx
```

窗口里可以打开文件，也可以把 `.onnx`、`.safetensors`、`.gguf`、`.tflite`、`.pt`、`.pb`、`.keras`、`.h5` 拖进去。应用名是 AI Model Viewer，bundle id 是 `dev.ai-model-viewer`，不会占用已经安装的 Netron.app。

张量权重数据块只记名字、dtype 和 shape，不把多 GB 的权重正文读进图里。

## 限制

- 静态布局遇到环时，环上的算子放在同一层，不再逐层展开。
- 纯权重检查点（如 `.safetensors`、`.gguf` 或仅含 `state_dict` 的 `.pt`）按张量命名前缀自动合成为层级模块图。
- 属性里的大张量和子图只显示类型或 `<graph>`。
