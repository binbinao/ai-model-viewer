// Draws the AI Model Viewer app icon: a three-node graph on a dark field.
// Run from the repo root: swift macos/render_icon.swift
import AppKit
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let canvas: CGFloat = 1024

struct Card {
    var center: CGPoint
    var size: CGSize
    var radius: CGFloat
    var top: (CGFloat, CGFloat, CGFloat)
    var bottom: (CGFloat, CGFloat, CGFloat)
}

let input = Card(
    center: CGPoint(x: 360, y: 800),
    size: CGSize(width: 300, height: 148),
    radius: 36,
    top: (0.49, 0.75, 1.0),
    bottom: (0.16, 0.42, 0.74)
)
let op = Card(
    center: CGPoint(x: 512, y: 512),
    size: CGSize(width: 332, height: 164),
    radius: 40,
    top: (0.72, 0.66, 1.0),
    bottom: (0.34, 0.27, 0.82)
)
let output = Card(
    center: CGPoint(x: 664, y: 224),
    size: CGSize(width: 300, height: 148),
    radius: 36,
    top: (0.45, 0.92, 0.70),
    bottom: (0.08, 0.55, 0.36)
)

func rgb(_ c: (CGFloat, CGFloat, CGFloat), _ a: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: c.0, green: c.1, blue: c.2, alpha: a)
}

func rect(of card: Card) -> CGRect {
    CGRect(
        x: card.center.x - card.size.width / 2,
        y: card.center.y - card.size.height / 2,
        width: card.size.width,
        height: card.size.height
    )
}

func draw(in ctx: CGContext, size: Int) {
    let scale = CGFloat(size) / canvas
    ctx.scaleBy(x: scale, y: scale)
    ctx.setAllowsAntialiasing(true)
    ctx.setShouldAntialias(true)
    ctx.interpolationQuality = .high

    let bounds = CGRect(x: 0, y: 0, width: canvas, height: canvas)
    ctx.saveGState()
    ctx.addRect(bounds)
    ctx.clip()
    let background = CGGradient(
        colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
        colors: [rgb((0.22, 0.28, 0.40)), rgb((0.07, 0.08, 0.11))] as CFArray,
        locations: [0, 1]
    )!
    ctx.drawRadialGradient(
        background,
        startCenter: CGPoint(x: 390, y: 760),
        startRadius: 20,
        endCenter: CGPoint(x: 512, y: 480),
        endRadius: 820,
        options: [.drawsAfterEndLocation]
    )
    ctx.restoreGState()

    ctx.saveGState()
    ctx.setStrokeColor(rgb((0.95, 0.96, 0.98)))
    ctx.setLineWidth(54)
    ctx.setLineCap(.round)
    ctx.setLineJoin(.round)
    let link = CGMutablePath()
    link.move(to: input.center)
    link.addCurve(
        to: op.center,
        control1: CGPoint(x: input.center.x, y: 640),
        control2: CGPoint(x: op.center.x, y: 670)
    )
    link.move(to: op.center)
    link.addCurve(
        to: output.center,
        control1: CGPoint(x: op.center.x, y: 360),
        control2: CGPoint(x: output.center.x, y: 390)
    )
    ctx.addPath(link)
    ctx.strokePath()
    ctx.restoreGState()

    for card in [input, op, output] {
        let box = rect(of: card)
        let path = CGPath(roundedRect: box, cornerWidth: card.radius, cornerHeight: card.radius, transform: nil)
        ctx.saveGState()
        ctx.setShadow(
            offset: CGSize(width: 0, height: -14),
            blur: 28,
            color: rgb((0, 0, 0), 0.45)
        )
        ctx.setFillColor(rgb(card.bottom))
        ctx.addPath(path)
        ctx.fillPath()
        ctx.restoreGState()

        ctx.saveGState()
        ctx.addPath(path)
        ctx.clip()
        let fill = CGGradient(
            colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
            colors: [rgb(card.top), rgb(card.bottom)] as CFArray,
            locations: [0, 1]
        )!
        ctx.drawLinearGradient(
            fill,
            start: CGPoint(x: box.midX, y: box.maxY),
            end: CGPoint(x: box.midX, y: box.minY),
            options: []
        )
        ctx.setFillColor(rgb((1, 1, 1), 0.22))
        ctx.fill(CGRect(x: box.minX, y: box.maxY - 46, width: box.width, height: 46))
        ctx.restoreGState()

        ctx.saveGState()
        ctx.addPath(path)
        ctx.setStrokeColor(rgb((1, 1, 1), 0.28))
        ctx.setLineWidth(6)
        ctx.strokePath()
        ctx.restoreGState()
    }
}

func render(_ size: Int) -> CGImage {
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let ctx = CGContext(
        data: nil,
        width: size,
        height: size,
        bitsPerComponent: 8,
        bytesPerRow: 0,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )!
    draw(in: ctx, size: size)
    return ctx.makeImage()!
}

func writePNG(_ image: CGImage, to url: URL) {
    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
        fputs("failed to write \(url.path)\n", stderr)
        exit(1)
    }
}

let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
let iconset = FileManager.default.temporaryDirectory.appendingPathComponent("ai-model-viewer-icon.iconset", isDirectory: true)
if FileManager.default.fileExists(atPath: iconset.path) {
    try FileManager.default.removeItem(at: iconset)
}
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
let sizes = [
    (16, "icon_16x16.png"),
    (32, "icon_16x16@2x.png"),
    (32, "icon_32x32.png"),
    (64, "icon_32x32@2x.png"),
    (128, "icon_128x128.png"),
    (256, "icon_128x128@2x.png"),
    (256, "icon_256x256.png"),
    (512, "icon_256x256@2x.png"),
    (512, "icon_512x512.png"),
    (1024, "icon_512x512@2x.png"),
]
var cache: [Int: CGImage] = [:]
for (size, name) in sizes {
    let image = cache[size] ?? render(size)
    cache[size] = image
    writePNG(image, to: iconset.appendingPathComponent(name))
}
writePNG(cache[1024]!, to: root.appendingPathComponent("AppIcon.png"))
let icns = root.appendingPathComponent("AppIcon.icns")
let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconset.path, "-o", icns.path]
try iconutil.run()
iconutil.waitUntilExit()
if iconutil.terminationStatus != 0 {
    fputs("iconutil failed\n", stderr)
    exit(iconutil.terminationStatus)
}
try FileManager.default.removeItem(at: iconset)
