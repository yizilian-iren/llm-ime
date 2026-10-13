import AppKit
import QuartzCore

/// The floating candidate window (N-008): glass (behind-window blur + the view's tint), rounded,
/// no border, never takes focus or clicks. One for the whole input method.
final class CandidatePanel {
    static let shared = CandidatePanel()

    private let panel: NSPanel
    private let glass: NSVisualEffectView
    private let view: CandidateView
    private var shownFor: NSRect?  // the cursor line it is placed against

    var appearance: Appearance {
        get { Appearance(rawValue: UserDefaults.standard.string(forKey: "appearance") ?? "") ?? .system }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "appearance"); applyTheme() }
    }

    private init() {
        panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 200, height: 60),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false
        panel.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.popUpMenuWindow)) + 1)
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]

        glass = NSVisualEffectView()
        glass.material = .popover
        glass.blendingMode = .behindWindow
        glass.state = .active
        glass.maskImage = Self.roundedMask(radius: CandidateView.corner)
        glass.autoresizingMask = [.width, .height]

        view = CandidateView()
        view.wantsLayer = true
        view.autoresizingMask = [.width, .height]
        glass.addSubview(view)
        panel.contentView = glass

        DistributedNotificationCenter.default().addObserver(
            forName: Notification.Name("AppleInterfaceThemeChangedNotification"), object: nil, queue: .main
        ) { [weak self] _ in self?.applyTheme() }
        applyTheme()
    }

    private static var systemDark: Bool {
        UserDefaults.standard.string(forKey: "AppleInterfaceStyle") == "Dark"
    }

    private func applyTheme() {
        let theme = appearance.theme(systemDark: Self.systemDark)
        view.theme = theme
        glass.appearance = NSAppearance(named: theme.dark ? .darkAqua : .aqua)
    }

    /// Shows `model` under the cursor line `line` (screen coordinates, as IMK reports it).
    /// `fade`: the candidates were replaced by a better answer for the same keys (quick → full).
    func show(_ model: CandidateModel, below line: NSRect, fade: Bool) {
        if model.candidates.isEmpty {
            if !panel.isVisible { return }  // nothing to show yet: wait for the first answer
        }
        if fade, let layer = view.layer {
            let t = CATransition()
            t.type = .fade
            t.duration = 0.16
            layer.add(t, forKey: "fade")
        }
        view.model = model
        let size = view.fittingSize()
        let frame = place(size, line: line)
        if panel.isVisible && shownFor == line {
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.12
                ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
                panel.animator().setFrame(frame, display: true)
            }
        } else {
            panel.setFrame(frame, display: true)
            panel.orderFrontRegardless()
        }
        shownFor = line
    }

    func hide() {
        panel.orderOut(nil)
        shownFor = nil
    }

    /// Left edge so the keys line up with the marked text; below the line, or above it near the bottom of
    /// the screen; pulled in from the screen's edges.
    private func place(_ size: NSSize, line: NSRect) -> NSRect {
        let screen = NSScreen.screens.first { $0.frame.contains(line.origin) } ?? NSScreen.main
        let vis = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        var x = line.minX - CandidateView.pad - CandidateView.itemInset.width
        var y = line.minY - 4 - size.height
        if y < vis.minY { y = line.maxY + 4 }
        x = min(max(x, vis.minX + 4), vis.maxX - size.width - 4)
        y = min(max(y, vis.minY + 4), vis.maxY - size.height - 4)
        return NSRect(origin: NSPoint(x: x, y: y), size: size)
    }

    private static func roundedMask(radius: CGFloat) -> NSImage {
        let d = radius * 2 + 1
        let img = NSImage(size: NSSize(width: d, height: d), flipped: false) { r in
            NSColor.black.setFill()
            NSBezierPath(roundedRect: r, xRadius: radius, yRadius: radius).fill()
            return true
        }
        img.capInsets = NSEdgeInsets(top: radius, left: radius, bottom: radius, right: radius)
        img.resizingMode = .stretch
        return img
    }
}
