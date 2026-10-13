import AppKit
import CoreImage

// Draws the candidate window over a fake document with made-up candidates, light and dark, to PNGs:
//   mock <out dir>
// The glass blur and the window shadow are imitated here (the real panel gets them from AppKit).

let scale: CGFloat = 2
let sceneSize = NSSize(width: 720, height: 300)

struct Scene {
    let name: String
    let theme: Theme
    let page, ink, faint: NSColor
}

let scenes = [
    Scene(name: "light", theme: .light, page: hex(0xffffff), ink: hex(0x1d1d1f), faint: hex(0x8a8a8e)),
    Scene(name: "dark", theme: .darkTheme, page: hex(0x1e1e1e), ink: hex(0xe6e6e6), faint: hex(0x8a8a8e)),
]

let model = CandidateModel(
    keys: "jintianqunachifan",
    candidates: ["今天去哪吃饭", "今天去那吃饭", "金天去哪吃饭", "今天去拿吃饭", "今天趣哪吃饭"],
    selected: 0)

func render(_ s: Scene, to url: URL) {
    let w = Int(sceneSize.width * scale), h = Int(sceneSize.height * scale)
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let cg = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    cg.translateBy(x: 0, y: CGFloat(h))
    cg.scaleBy(x: scale, y: -scale)
    let ns = NSGraphicsContext(cgContext: cg, flipped: true)
    NSGraphicsContext.current = ns

    // The document: a few lines, the last one ending in the marked keys.
    s.page.setFill()
    NSRect(origin: .zero, size: sceneSize).fill()
    let body = NSFont.systemFont(ofSize: 15)
    let lines = [
        "这周的实验跑完了，结果比上周好一点，",
        "周五想和大家一起吃个饭，地方还没定。",
        "下班以后，",
    ]
    var y: CGFloat = 36
    let lineH: CGFloat = 26
    for l in lines {
        NSAttributedString(string: l, attributes: [.font: body, .foregroundColor: s.ink]).draw(at: NSPoint(x: 40, y: y))
        y += lineH
    }
    let caretLine = y - lineH
    let prefixW = NSAttributedString(string: lines.last!, attributes: [.font: body]).size().width
    let marked = NSAttributedString(string: model.keys, attributes: [
        .font: body, .foregroundColor: s.ink, .underlineStyle: NSUnderlineStyle.single.rawValue,
        .underlineColor: s.faint])
    marked.draw(at: NSPoint(x: 40 + prefixW, y: caretLine))
    for (i, l) in ["大家有想去的地方可以在群里说一下，", "我周四晚上统计好发给大家。"].enumerated() {
        NSAttributedString(string: l, attributes: [.font: body, .foregroundColor: s.ink])
            .draw(at: NSPoint(x: 40, y: caretLine + lineH * CGFloat(i + 1) + 34))
    }
    ns.flushGraphics()
    let backdrop = cg.makeImage()!

    // The panel, just under the marked text.
    let view = CandidateView(frame: .zero)
    view.theme = s.theme
    view.model = model
    let size = view.fittingSize()
    view.frame = NSRect(origin: .zero, size: size)
    let origin = NSPoint(x: 40 + prefixW - CandidateView.pad - CandidateView.itemInset.width, y: caretLine + lineH + 4)
    let frame = NSRect(origin: origin, size: size)
    let path = NSBezierPath(roundedRect: frame, xRadius: CandidateView.corner, yRadius: CandidateView.corner)

    // Deck's --shadow-float: 0 0 0 .5px 20%, 0 16px 48px 22%, 0 4px 12px 8% (black).
    for (dy, blur, a) in [(16.0, 48.0, 0.22), (4.0, 12.0, 0.08), (0.0, 1.0, 0.20)] as [(CGFloat, CGFloat, CGFloat)] {
        NSGraphicsContext.saveGraphicsState()
        let sh = NSShadow()
        sh.shadowOffset = NSSize(width: 0, height: -dy * scale)
        sh.shadowBlurRadius = blur * scale / 2
        sh.shadowColor = NSColor.black.withAlphaComponent(s.theme.dark ? min(a * 1.5, 0.35) : a)
        sh.set()
        s.theme.base.setFill()
        path.fill()
        NSGraphicsContext.restoreGraphicsState()
    }

    // Glass: the document behind the panel, blurred (behind-window material), then the tint (in the view).
    let ci = CIImage(cgImage: backdrop)
    let blurred = ci.clampedToExtent()
        .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 22 * scale])
        .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 1.6])
        .cropped(to: ci.extent)
    let blurImg = CIContext().createCGImage(blurred, from: ci.extent)!
    NSGraphicsContext.saveGraphicsState()
    path.addClip()
    cg.saveGState()
    cg.translateBy(x: 0, y: sceneSize.height)
    cg.scaleBy(x: 1, y: -1)
    cg.draw(blurImg, in: CGRect(origin: .zero, size: sceneSize))
    cg.restoreGState()
    cg.translateBy(x: origin.x, y: origin.y)
    view.draw(view.bounds)
    NSGraphicsContext.restoreGraphicsState()

    ns.flushGraphics()
    let rep = NSBitmapImageRep(cgImage: cg.makeImage()!)
    rep.size = sceneSize
    try! rep.representation(using: .png, properties: [:])!.write(to: url)
    print(url.path, size)
}

let out = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : ".")
try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
for s in scenes {
    render(s, to: out.appendingPathComponent("candidates-\(s.name).png"))
}
