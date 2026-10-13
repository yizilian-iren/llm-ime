import AppKit

// The menu bar icon (a template: macOS tints it): 言 in a rounded square, 16 pt at 1x and 2x, to a TIFF.
//   icon <out.tiff>

func rep(_ scale: CGFloat) -> NSBitmapImageRep {
    let px = Int(16 * scale)
    let r = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8,
                             samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                             bytesPerRow: 0, bitsPerPixel: 0)!
    r.size = NSSize(width: 16, height: 16)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: r)
    let box = NSRect(x: 1, y: 1, width: 14, height: 14)
    NSColor.black.setFill()
    NSBezierPath(roundedRect: box, xRadius: 3.5, yRadius: 3.5).fill()
    let s = NSAttributedString(string: "言", attributes: [
        .font: NSFont.systemFont(ofSize: 11, weight: .semibold), .foregroundColor: NSColor.black])
    let sz = s.size()
    NSGraphicsContext.current?.compositingOperation = .destinationOut
    s.draw(at: NSPoint(x: 8 - sz.width / 2, y: 8 - sz.height / 2 + 0.2))
    NSGraphicsContext.restoreGraphicsState()
    return r
}

let img = NSImage(size: NSSize(width: 16, height: 16))
img.addRepresentation(rep(1))
img.addRepresentation(rep(2))
try! img.tiffRepresentation!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
