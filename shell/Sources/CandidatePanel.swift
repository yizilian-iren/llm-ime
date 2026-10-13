import AppKit
import QuartzCore

/// The floating candidate window (N-008): glass (behind-window blur + the view's tint), rounded,
/// no border, never takes focus or clicks. One for the whole input method.
/// Hidden settings for trying the glass on a real screen, read each time the window appears:
///   defaults write com.iren.inputmethod.LLMIME tint -float 0.4        the tint's share of base (0-1; 0.6)
///   defaults write com.iren.inputmethod.LLMIME material hudWindow     the blur's material, by name; none: no blur
/// (defaults delete ... to go back).
final class CandidatePanel: NSObject {
    static let shared = CandidatePanel()

    private let panel: NSPanel
    private let glass: NSVisualEffectView
    private let view: CandidateView
    private let shadow: ShadowView
    private var shownFor: NSRect?  // the cursor line it is placed against

    var appearance: Appearance {
        get { Appearance(rawValue: UserDefaults.standard.string(forKey: "appearance") ?? "") ?? .system }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "appearance"); applyTheme() }
    }

    private static let materials: [String: NSVisualEffectView.Material] = [
        "popover": .popover, "menu": .menu, "hudWindow": .hudWindow, "underWindowBackground": .underWindowBackground,
        "fullScreenUI": .fullScreenUI, "sidebar": .sidebar, "headerView": .headerView, "sheet": .sheet,
        "windowBackground": .windowBackground, "toolTip": .toolTip, "titlebar": .titlebar,
        "contentBackground": .contentBackground, "underPageBackground": .underPageBackground,
    ]
    /// The material from the hidden setting; nil for "none" (no blur: the tint alone).
    private static var material: NSVisualEffectView.Material? {
        let name = UserDefaults.standard.string(forKey: "material") ?? "popover"
        return name == "none" ? nil : materials[name] ?? .popover
    }
    private static var tintAlpha: CGFloat {
        let d = UserDefaults.standard
        let a = d.object(forKey: "tint") == nil ? 0.60 : d.double(forKey: "tint")  // -float or a plain string
        return CGFloat(min(max(a, 0), 1))
    }

    private override init() {
        let m = ShadowView.margin
        let card = NSRect(x: m.left, y: m.bottom, width: 200, height: 60)
        let panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: card.maxX + m.right, height: card.maxY + m.top),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false  // the system's shadow brings a 0.5 pt dark line around the window: ShadowView instead
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false
        panel.level = NSWindow.Level(rawValue: Int(CGWindowLevelForKey(.popUpMenuWindow)) + 1)
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]

        let glass = NSVisualEffectView(frame: NSRect(origin: .zero, size: card.size))
        glass.material = .popover
        glass.blendingMode = .behindWindow
        glass.state = .active
        glass.maskImage = Self.roundedMask(radius: CandidateView.corner)
        glass.autoresizingMask = [.width, .height]

        let view = CandidateView(frame: NSRect(origin: .zero, size: card.size))
        view.wantsLayer = true
        view.autoresizingMask = [.width, .height]
        let cardView = NSView(frame: card)  // the glass under the view, as siblings, so "none" can hide the glass alone
        cardView.autoresizingMask = [.width, .height]
        cardView.addSubview(glass)
        cardView.addSubview(view)
        let shadow = ShadowView(frame: NSRect(origin: .zero, size: panel.frame.size))
        shadow.autoresizingMask = [.width, .height]
        let content = NSView(frame: shadow.frame)
        content.addSubview(shadow)
        content.addSubview(cardView)
        panel.contentView = content
        (self.panel, self.glass, self.view, self.shadow) = (panel, glass, view, shadow)
        super.init()

        // The input method is never the active app, and distributed notifications to inactive apps are held
        // back by default: ask for them at once. show() also reads the appearance each time, in case one is missed.
        DistributedNotificationCenter.default().addObserver(
            self, selector: #selector(themeChanged), name: Notification.Name("AppleInterfaceThemeChangedNotification"),
            object: nil, suspensionBehavior: .deliverImmediately)
        applyTheme()
    }

    @objc private func themeChanged(_ note: Notification) {
        DispatchQueue.main.async { self.applyTheme() }
    }

    /// The system's light / dark as AppKit draws it. Not the AppleInterfaceStyle default: on M3 (macOS 27) the
    /// system was dark with that key absent. AppKit updates this on the theme notification, before ours arrives.
    private static var systemDark: Bool {
        NSApp.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
    }

    private func applyTheme() {
        var theme = appearance.theme(systemDark: appearance == .system && Self.systemDark)
        theme.tintAlpha = Self.tintAlpha
        let material = Self.material
        glass.isHidden = material == nil
        if let m = material, glass.material != m { glass.material = m }
        if view.theme.dark == theme.dark && view.theme.tintAlpha == theme.tintAlpha && glass.appearance != nil { return }
        view.theme = theme
        shadow.dark = theme.dark
        glass.appearance = NSAppearance(named: theme.dark ? .darkAqua : .aqua)
    }

    /// Shows `model` under the cursor line `line` (screen coordinates, as IMK reports it).
    /// `fade`: the candidates were replaced by a better answer for the same keys (quick → full).
    func show(_ model: CandidateModel, below line: NSRect, fade: Bool) {
        if model.candidates.isEmpty {
            if !panel.isVisible { return }  // nothing to show yet: wait for the first answer
        }
        if !panel.isVisible { applyTheme() }
        if fade, let layer = view.layer {
            let t = CATransition()
            t.type = .fade
            t.duration = 0.16
            layer.add(t, forKey: "fade")
        }
        view.model = model
        let size = view.fittingSize()
        let m = ShadowView.margin
        let card = place(size, line: line)
        let frame = NSRect(x: card.minX - m.left, y: card.minY - m.bottom,
                           width: card.width + m.left + m.right, height: card.height + m.bottom + m.top)
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

/// The window's shadow, drawn in a transparent margin around the card: Deck's --shadow-float (0 16px 48px 22%,
/// 0 4px 12px 8% black) without its 0.5px ring (user: 「去掉吧」). Only outside the card, so the glass is not darkened.
final class ShadowView: NSView {
    static let margin = NSEdgeInsets(top: 24, left: 36, bottom: 56, right: 36)
    private static let layers: [(dy: CGFloat, blur: CGFloat, alpha: CGFloat)] = [(16, 24, 0.22), (4, 6, 0.08)]

    var dark = false { didSet { if dark != oldValue { needsDisplay = true } } }

    override func setFrameSize(_ size: NSSize) {
        super.setFrameSize(size)
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        let m = Self.margin
        let card = NSRect(x: m.left, y: m.bottom, width: bounds.width - m.left - m.right,
                          height: bounds.height - m.bottom - m.top)
        let path = NSBezierPath(roundedRect: card, xRadius: CandidateView.corner, yRadius: CandidateView.corner)
        let outside = NSBezierPath(rect: bounds)
        outside.append(path)
        outside.windingRule = .evenOdd
        outside.addClip()
        for l in Self.layers {
            NSGraphicsContext.saveGraphicsState()
            let sh = NSShadow()
            sh.shadowOffset = NSSize(width: 0, height: -l.dy)
            sh.shadowBlurRadius = l.blur
            sh.shadowColor = NSColor.black.withAlphaComponent(dark ? min(l.alpha * 1.5, 0.35) : l.alpha)
            sh.set()
            NSColor.black.setFill()
            path.fill()  // clipped away: only its shadow shows
            NSGraphicsContext.restoreGraphicsState()
        }
    }
}
