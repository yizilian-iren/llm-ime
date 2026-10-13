import AppKit

/// What the candidate window shows: whole-sentence candidates, five to a page (the keys are only the app's
/// marked text). `selected` counts from the first candidate of all; the page is the one holding it.
struct CandidateModel: Equatable {
    static let pageSize = 5
    var candidates: [String] = []
    var selected = 0

    var page: Int { selected / Self.pageSize }
    var pageRange: Range<Int> {
        let start = page * Self.pageSize
        return start..<min(start + Self.pageSize, candidates.count)
    }
    var hasPrev: Bool { page > 0 }
    var hasNext: Bool { pageRange.upperBound < candidates.count }
    var current: String? { candidates.indices.contains(selected) ? candidates[selected] : nil }

    /// The candidate in slot `n` (1-5) of this page.
    func slot(_ n: Int) -> String? {
        let i = pageRange.lowerBound + n - 1
        return pageRange.contains(i) ? candidates[i] : nil
    }

    mutating func turn(_ step: Int) {
        let p = page + step
        if p >= 0 && p * Self.pageSize < candidates.count { selected = p * Self.pageSize }
    }

    mutating func move(_ step: Int) {
        selected = min(max(selected + step, 0), max(candidates.count - 1, 0))
    }
}

/// Draws the candidate window's content (N-008): the candidates in one row, nothing else.
/// It paints the glass tint itself; the blur under it is the window's NSVisualEffectView.
final class CandidateView: NSView {
    static let corner: CGFloat = 10
    static let pad: CGFloat = 2             // panel edge to the items: the selected one fills the window but this
    static let itemInset = NSSize(width: 6, height: 4)
    static let itemCorner: CGFloat = corner - pad  // concentric with the window's corners
    static let gap: CGFloat = 1             // between two items
    static let arrowsWidth: CGFloat = 22     // the faint ‹ › after the items when there are more pages

    var theme = Theme.light { didSet { needsDisplay = true } }
    var model = CandidateModel() { didSet { needsDisplay = true } }

    override var isFlipped: Bool { true }

    private static let candFont = NSFont.systemFont(ofSize: 15)
    private static let indexFont = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .medium)

    /// Slot `i` (0-4) of the current page.
    private func item(_ i: Int) -> NSAttributedString {
        let index = model.pageRange.lowerBound + i
        let s = NSMutableAttributedString(string: "\(i + 1)", attributes: [
            .font: Self.indexFont, .kern: 4,
            .foregroundColor: index == model.selected ? theme.mauve : theme.subtext0])
        s.append(NSAttributedString(string: model.candidates[index], attributes: [
            .font: Self.candFont, .foregroundColor: theme.text]))
        return s
    }

    private var itemHeight: CGFloat {
        ceil(item0Height) + Self.itemInset.height * 2
    }
    private var item0Height: CGFloat {
        NSAttributedString(string: "1中", attributes: [.font: Self.candFont]).size().height
    }

    private var paged: Bool { model.hasPrev || model.hasNext }

    /// The rectangles of this page's items, left to right, in view coordinates.
    func itemRects() -> [NSRect] {
        var x = Self.pad
        let y = Self.pad
        return (0..<model.pageRange.count).map { i in
            let w = ceil(item(i).size().width) + Self.itemInset.width * 2
            defer { x += w + Self.gap }
            return NSRect(x: x, y: y, width: w, height: itemHeight)
        }
    }

    /// The size the window should have for the current model.
    func fittingSize() -> NSSize {
        let rects = itemRects()
        let itemsW = rects.last.map { $0.maxX - Self.pad } ?? 0
        let w = max(itemsW, 60) + Self.pad * 2 + (paged ? Self.arrowsWidth : 0)
        let h = Self.pad + itemHeight + Self.pad
        return NSSize(width: ceil(w), height: ceil(h))
    }

    override func draw(_ dirtyRect: NSRect) {
        theme.tint.setFill()
        NSBezierPath(roundedRect: bounds, xRadius: Self.corner, yRadius: Self.corner).fill()

        let rects = itemRects()
        for (i, r) in rects.enumerated() {
            if model.pageRange.lowerBound + i == model.selected {
                theme.selected.setFill()
                NSBezierPath(roundedRect: r, xRadius: Self.itemCorner, yRadius: Self.itemCorner).fill()
            }
            item(i).draw(at: NSPoint(x: r.minX + Self.itemInset.width, y: r.minY + Self.itemInset.height))
        }
        if paged, let last = rects.last {
            drawArrows(in: NSRect(x: last.maxX, y: last.minY, width: Self.arrowsWidth, height: last.height))
        }
    }

    /// ‹ › centered in `r`; the side that has no page is fainter still.
    private func drawArrows(in r: NSRect) {
        let font = NSFont.systemFont(ofSize: 13, weight: .medium)
        for (i, (glyph, on)) in [("‹", model.hasPrev), ("›", model.hasNext)].enumerated() {
            let a = NSAttributedString(string: glyph, attributes: [
                .font: font, .foregroundColor: theme.overlay1.withAlphaComponent(on ? 1 : 0.35)])
            let sz = a.size()
            let cx = r.midX + (i == 0 ? -5 : 5)
            a.draw(at: NSPoint(x: cx - sz.width / 2, y: r.midY - sz.height / 2))
        }
    }
}
