import AppKit

/// The candidate window's colors: Agent Deck's Warm tone (warmLight / warmDark of its catppuccin.ts), N-008.
struct Theme {
    let base, mantle, surface0, text, subtext0, overlay1, overlay2, mauve: NSColor
    let dark: Bool

    static let light = Theme(
        base: hex(0xfaf9f5), mantle: hex(0xf3f1ea), surface0: hex(0xe6e2d7), text: hex(0x3d3a33),
        subtext0: hex(0x6e685c), overlay1: hex(0x968f80), overlay2: hex(0x847d6e), mauve: hex(0x8839ef), dark: false)

    static let darkTheme = Theme(
        base: hex(0x262624), mantle: hex(0x1f1e1c), surface0: hex(0x33322f), text: hex(0xe8e3d6),
        subtext0: hex(0xada799), overlay1: hex(0x7f7b72), overlay2: hex(0x959085), mauve: hex(0xcba6f7), dark: true)

    /// How much of base the glass tint is: 60%, so the page shows through (user, rv_mKRBScBm).
    var tintAlpha: CGFloat = 0.60
    /// The glass tint laid over the blur.
    var tint: NSColor { base.withAlphaComponent(tintAlpha) }
    /// The selected candidate's background (Deck's --selected: overlay2 at 20%).
    var selected: NSColor { overlay2.withAlphaComponent(0.20) }
}

/// The user's choice in the input menu; `.system` follows macOS's appearance like Deck's theme "system".
enum Appearance: String {
    case system, light, dark

    func theme(systemDark: Bool) -> Theme {
        switch self {
        case .system: return systemDark ? .darkTheme : .light
        case .light: return .light
        case .dark: return .darkTheme
        }
    }
}

func hex(_ v: UInt32) -> NSColor {
    NSColor(srgbRed: CGFloat((v >> 16) & 0xff) / 255, green: CGFloat((v >> 8) & 0xff) / 255,
            blue: CGFloat(v & 0xff) / 255, alpha: 1)
}
