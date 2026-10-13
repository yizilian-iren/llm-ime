import AppKit

/// What the candidate window shows: up to five whole-sentence candidates (the keys are only the app's marked text).
struct CandidateModel: Equatable {
    var candidates: [String] = []
    var selected = 0
}

/// Draws the candidate window's content (N-008): the candidates in one row, nothing else.
/// It paints the glass tint itself; the blur under it is the window's NSVisualEffectView.
final class CandidateView: NSView {
    static let corner: CGFloat = 10
    static let pad: CGFloat = 4             // panel edge to the items
    static let itemInset = NSSize(width: 8, height: 5)
    static let itemCorner: CGFloat = 6
    static let gap: CGFloat = 2             // between two items

    var theme = Theme.light { didSet { needsDisplay = true } }
    var model = CandidateModel() { didSet { needsDisplay = true } }

    override var isFlipped: Bool { true }

    private static let candFont = NSFont.systemFont(ofSize: 15)
    private static let indexFont = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .medium)

    private func item(_ i: Int) -> NSAttributedString {
        let s = NSMutableAttributedString(string: "\(i + 1)", attributes: [
            .font: Self.indexFont, .kern: 5,
            .foregroundColor: i == model.selected ? theme.mauve : theme.subtext0])
        s.append(NSAttributedString(string: model.candidates[i], attributes: [
            .font: Self.candFont, .foregroundColor: theme.text]))
        return s
    }

    private var itemHeight: CGFloat {
        ceil(item0Height) + Self.itemInset.height * 2
    }
    private var item0Height: CGFloat {
        NSAttributedString(string: "1中", attributes: [.font: Self.candFont]).size().height
    }

    /// The item rectangles, left to right, in view coordinates.
    func itemRects() -> [NSRect] {
        var x = Self.pad
        let y = Self.pad
        return model.candidates.indices.map { i in
            let w = ceil(item(i).size().width) + Self.itemInset.width * 2
            defer { x += w + Self.gap }
            return NSRect(x: x, y: y, width: w, height: itemHeight)
        }
    }

    /// The size the window should have for the current model.
    func fittingSize() -> NSSize {
        let rects = itemRects()
        let itemsW = rects.last.map { $0.maxX - Self.pad } ?? 0
        let w = max(itemsW, 60) + Self.pad * 2
        let h = Self.pad + itemHeight + Self.pad
        return NSSize(width: ceil(w), height: ceil(h))
    }

    override func draw(_ dirtyRect: NSRect) {
        theme.tint.setFill()
        NSBezierPath(roundedRect: bounds, xRadius: Self.corner, yRadius: Self.corner).fill()

        for (i, r) in itemRects().enumerated() {
            if i == model.selected {
                theme.selected.setFill()
                NSBezierPath(roundedRect: r, xRadius: Self.itemCorner, yRadius: Self.itemCorner).fill()
            }
            item(i).draw(at: NSPoint(x: r.minX + Self.itemInset.width, y: r.minY + Self.itemInset.height))
        }
    }
}
