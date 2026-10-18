import AppKit
import CoreImage

// Draws the candidate window over a fake note with a colored picture, light and dark, to PNGs, with the menu
// bar icon in a fake menu bar:
//   mock <out dir> [icon.tiff]
// The glass blur and the window shadow are imitated here (the real panel gets them from AppKit).

let scale: CGFloat = 2
let sceneSize = NSSize(width: 780, height: 340)

struct Scene {
    let name: String
    let theme: Theme
    let page, ink, faint, bar: NSColor
}

let scenes = [
    Scene(name: "light", theme: .light, page: hex(0xffffff), ink: hex(0x1d1d1f), faint: hex(0x8a8a8e), bar: hex(0xebe8e1)),
    Scene(name: "dark", theme: .darkTheme, page: hex(0x1e1e1e), ink: hex(0xe6e6e6), faint: hex(0x8a8a8e), bar: hex(0x2b2a27)),
]

let shots: [(name: String, prefix: String, keys: String, model: CandidateModel)] = [
    ("candidates", "下班以后，", "jintianqunachifan", CandidateModel(
        candidates: ["今天去哪吃饭", "今天去那吃饭", "今天去拿吃饭", "今天去哪吃饭呢", "今天去那吃番",
                     "今天去那吃饭呢", "今天去哪吃番", "今天去那吃饭那", "今天去拿吃饭呢", "今天去哪吃饭吧", "今天去那吃饭吧"],
        selected: 0)),
    // 6B.2: the English mode, a misspelling: its correction first, then the keys as typed and the rest
    ("english", "It is not ", "neccesary", CandidateModel(
        candidates: ["necessary", "neccesary", "necessarily", "necessaries", "necessitarian", "necessity"], selected: 0)),
]
let icon = CommandLine.arguments.count > 2 ? NSImage(contentsOfFile: CommandLine.arguments[2]) : nil

/// A sunset picture: sky gradient, sun, two hills. Strong colors, so the glass's see-through shows.
func drawPicture(in r: NSRect) {
    NSGraphicsContext.saveGraphicsState()
    NSBezierPath(roundedRect: r, xRadius: 10, yRadius: 10).addClip()
    NSGradient(colors: [hex(0x6d597a), hex(0xe56b6f), hex(0xf4a259)])!.draw(in: r, angle: -90)
    hex(0xffd166).setFill()
    NSBezierPath(ovalIn: NSRect(x: r.minX + 300, y: r.minY + 60, width: 110, height: 110)).fill()
    for (dx, h, c) in [(-80.0, 120.0, hex(0x2a9d8f)), (260.0, 90.0, hex(0x264653))] as [(CGFloat, CGFloat, NSColor)] {
        c.setFill()
        NSBezierPath(ovalIn: NSRect(x: r.minX + dx, y: r.maxY - h, width: 560, height: h * 2)).fill()
    }
    NSGraphicsContext.restoreGraphicsState()
}

func text(_ s: String, _ font: NSFont, _ color: NSColor) -> NSAttributedString {
    NSAttributedString(string: s, attributes: [.font: font, .foregroundColor: color])
}

func render(_ s: Scene, _ shot: (name: String, prefix: String, keys: String, model: CandidateModel), to url: URL) {
    let (keys, model) = (shot.keys, shot.model)
    let w = Int(sceneSize.width * scale), h = Int(sceneSize.height * scale)
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let cg = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    cg.translateBy(x: 0, y: CGFloat(h))
    cg.scaleBy(x: scale, y: -scale)
    let ns = NSGraphicsContext(cgContext: cg, flipped: true)
    NSGraphicsContext.current = ns

    s.page.setFill()
    NSRect(origin: .zero, size: sceneSize).fill()

    // Menu bar with the input method's icon (a template image: drawn in the bar's text color).
    s.bar.setFill()
    NSRect(x: 0, y: 0, width: sceneSize.width, height: 26).fill()
    let barFont = NSFont.systemFont(ofSize: 13)
    var x: CGFloat = 16
    for (i, t) in ["备忘录", "文件", "编辑", "显示"].enumerated() {
        let a = text(t, i == 0 ? .boldSystemFont(ofSize: 13) : barFont, s.ink)
        a.draw(at: NSPoint(x: x, y: 5))
        x += a.size().width + 18
    }
    let clock = text("10月13日 周一 09:41", barFont, s.ink)
    clock.draw(at: NSPoint(x: sceneSize.width - 16 - clock.size().width, y: 5))
    if let icon = icon {
        let r = NSRect(x: sceneSize.width - 16 - clock.size().width - 30, y: 5, width: 16, height: 16)
        let tinted = NSImage(size: r.size, flipped: false) { b in
            icon.draw(in: b)
            s.ink.set()
            b.fill(using: .sourceAtop)
            return true
        }
        tinted.draw(in: r, from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
    }

    // The note: a title, the line being typed (marked keys), a picture under it.
    let body = NSFont.systemFont(ofSize: 15)
    text("周五聚餐", .boldSystemFont(ofSize: 20), s.ink).draw(at: NSPoint(x: 40, y: 48))
    let caretLine: CGFloat = 86, lineH: CGFloat = 26
    let prefix = text(shot.prefix, body, s.ink)
    prefix.draw(at: NSPoint(x: 40, y: caretLine))
    NSAttributedString(string: keys, attributes: [
        .font: body, .foregroundColor: s.ink, .underlineStyle: NSUnderlineStyle.single.rawValue,
        .underlineColor: s.faint]).draw(at: NSPoint(x: 40 + prefix.size().width, y: caretLine))
    drawPicture(in: NSRect(x: 40, y: caretLine + lineH + 10, width: sceneSize.width - 80, height: 190))
    ns.flushGraphics()
    let backdrop = cg.makeImage()!

    // The panel, just under the marked text, its first candidate lined up with the keys.
    let view = CandidateView(frame: .zero)
    view.theme = s.theme
    view.model = model
    let size = view.fittingSize()
    view.frame = NSRect(origin: .zero, size: size)
    let origin = NSPoint(x: 40 + prefix.size().width - CandidateView.pad - CandidateView.itemInset.width,
                         y: caretLine + lineH + 4)
    let frame = NSRect(origin: origin, size: size)
    let path = NSBezierPath(roundedRect: frame, xRadius: CandidateView.corner, yRadius: CandidateView.corner)

    // As ShadowView: Deck's --shadow-float without its 0.5px ring, 0 16px 48px 22%, 0 4px 12px 8% (black).
    // Drawn as a ring around the panel only, so it does not darken what shows through the glass.
    NSGraphicsContext.saveGraphicsState()
    let outside = NSBezierPath(rect: NSRect(origin: .zero, size: sceneSize))
    outside.append(path)
    outside.windingRule = .evenOdd
    outside.addClip()
    for (dy, blur, a) in [(16.0, 48.0, 0.22), (4.0, 12.0, 0.08)] as [(CGFloat, CGFloat, CGFloat)] {
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
    NSGraphicsContext.restoreGraphicsState()

    // Glass: what is behind the panel, blurred (behind-window material), then the view paints its tint.
    let ci = CIImage(cgImage: backdrop)
    let blurred = ci.clampedToExtent()
        .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 12 * scale])
        .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 1.4])
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
    for shot in shots {
        render(s, shot, to: out.appendingPathComponent("\(shot.name)-\(s.name).png"))
    }
}
