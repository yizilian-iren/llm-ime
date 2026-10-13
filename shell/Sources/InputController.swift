import AppKit
import InputMethodKit

/// One per text client. Letters build the key string (shown as marked text); the engine service answers
/// each change with a quick guess, then the full search, shown in the candidate panel. Whole sentences only.
@objc(LLMInputController)
final class InputController: IMKInputController {
    private static var english = false      // Shift toggles; shared by every client
    private static var history = ""         // what this input method committed, for apps that hide their text

    private var keys = ""
    private var context: String?             // text before the cursor when this composition began
    private var model = CandidateModel()
    private var requestId = 0
    private var stage = ""
    private var wantNextPage = false         // asked for the next page before the engine's further pages came
    private var shiftAlone = false

    override init!(server: IMKServer!, delegate: Any!, client inputClient: Any!) {
        super.init(server: server, delegate: delegate, client: inputClient)
    }

    override func recognizedEvents(_ sender: Any!) -> Int {
        Int(NSEvent.EventTypeMask([.keyDown, .flagsChanged]).rawValue)
    }

    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event = event, let client = sender as? IMKTextInput else { return false }
        if event.type == .flagsChanged {
            return flagsChanged(event, client)
        }
        guard event.type == .keyDown else { return false }
        shiftAlone = false
        if Self.english { return false }

        let mods = event.modifierFlags.intersection([.command, .control, .option])
        if !mods.isEmpty {
            if !keys.isEmpty { commit(keys, client) }  // a shortcut: leave the app nothing half-typed
            return false
        }
        let chars = event.characters ?? ""
        let composing = !keys.isEmpty

        switch event.keyCode {
        case 51:  // delete
            guard composing else { return false }
            keys.removeLast()
            keys.isEmpty ? reset(client) : update(client)
            return true
        case 53:  // esc
            guard composing else { return false }
            reset(client)
            return true
        case 36, 76:  // return, enter: the letters as typed
            guard composing else { return false }
            commit(keys, client)
            return true
        case 49:  // space
            guard composing else { return false }
            commit(model.current ?? keys, client)
            return true
        case 123, 124:  // left / right move the selection, on into the next or previous page
            guard composing else { return false }
            model.move(event.keyCode == 123 ? -1 : 1)
            show(client, fade: false)
            return true
        case 125, 121, 126, 116:  // down / page down, up / page up turn the page
            guard composing else { return false }
            turnPage(event.keyCode == 125 || event.keyCode == 121 ? 1 : -1, client)
            return true
        default:
            break
        }

        if chars.count == 1, let c = chars.unicodeScalars.first {
            if c.properties.isLowercase && c.isASCII {
                if !composing && !EngineClient.shared.connected { return false }  // no engine: type letters
                if !composing { context = textBeforeCursor(client) }
                keys.append(Character(c))
                update(client)
                return true
            }
            if composing, let n = Int(chars), (1...CandidateModel.pageSize).contains(n) {
                if let c = model.slot(n) { commit(c, client) }
                return true
            }
            if composing && (chars == "=" || chars == "-") {  // = / - turn the page
                turnPage(chars == "=" ? 1 : -1, client)
                return true
            }
            if let p = Punctuation.chinese(for: chars, after: composing ? nil : lastChar(client)) {
                if composing { commitBest(client) }
                insert(p, client)
                return true
            }
        }
        if composing { commitBest(client) }
        return false
    }

    /// Shift pressed and released with nothing in between toggles Chinese / English.
    private func flagsChanged(_ event: NSEvent, _ client: IMKTextInput) -> Bool {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if flags == .shift {
            shiftAlone = true
        } else if flags.isEmpty && shiftAlone {
            shiftAlone = false
            Self.english.toggle()
            if Self.english && !keys.isEmpty { commit(keys, client) }
        } else {
            shiftAlone = false
        }
        return false
    }

    // MARK: composing

    private func update(_ client: IMKTextInput) {
        client.setMarkedText(keys, selectionRange: NSRange(location: keys.utf16.count, length: 0),
                             replacementRange: NSRange(location: NSNotFound, length: 0))
        wantNextPage = false
        let ctx = context ?? String(Self.history.suffix(64))
        requestId = EngineClient.shared.type(context: ctx, keys: keys)
        if model.candidates.isEmpty { return }  // the panel appears with the first answer
        show(client, fade: false)
    }

    /// The engine answers each key with stage "quick", then "full" (five), then after a pause "more": the full
    /// answer's five followed by further candidates, which only add pages (the page on screen stays as it is).
    func engineAnswered(id: Int, stage: String, candidates: [String]) {
        guard id == requestId, !keys.isEmpty, let client = self.client() else { return }
        if stage == "more" {
            guard self.stage == "full" else { return }
            self.stage = stage
            model.candidates = candidates
            if wantNextPage {
                wantNextPage = false
                model.turn(1)
            }
            show(client, fade: false)
            return
        }
        let fade = stage == "full" && self.stage == "quick" && !model.candidates.isEmpty
        self.stage = stage
        model.candidates = candidates
        model.selected = 0
        show(client, fade: fade)
    }

    /// Turns the page; a next page the engine has not sent yet is turned to when it comes.
    private func turnPage(_ step: Int, _ client: IMKTextInput) {
        if step > 0 && !model.hasNext && stage == "full" {
            wantNextPage = true
            return
        }
        model.turn(step)
        show(client, fade: false)
    }

    private func show(_ client: IMKTextInput, fade: Bool) {
        CandidatePanel.shared.show(model, below: cursorLine(client), fade: fade)
    }

    private func commitBest(_ client: IMKTextInput) {
        commit(model.current ?? keys, client)
    }

    private func commit(_ text: String, _ client: IMKTextInput) {
        insert(text, client)
        if text != keys { EngineClient.shared.commit(text) }
        reset(client, clearMarked: false)
    }

    private func insert(_ text: String, _ client: IMKTextInput) {
        client.insertText(text, replacementRange: NSRange(location: NSNotFound, length: 0))
        Self.history = String((Self.history + text).suffix(64))
    }

    private func reset(_ client: IMKTextInput, clearMarked: Bool = true) {
        if clearMarked {
            client.setMarkedText("", selectionRange: NSRange(location: 0, length: 0),
                                 replacementRange: NSRange(location: NSNotFound, length: 0))
        }
        keys = ""
        context = nil
        model = CandidateModel()
        requestId = 0
        stage = ""
        wantNextPage = false
        CandidatePanel.shared.hide()
    }

    // MARK: reading the app's text

    /// Up to 64 characters before the cursor, or nil when the app does not tell (terminals, some Electron apps).
    private func textBeforeCursor(_ client: IMKTextInput) -> String? {
        let sel = client.selectedRange()
        guard sel.location != NSNotFound, sel.location > 0 else { return sel.location == 0 ? "" : nil }
        let start = max(0, sel.location - 64)
        let s = client.attributedSubstring(from: NSRange(location: start, length: sel.location - start))?.string
        return s.map { String($0.suffix(64)) }
    }

    private func lastChar(_ client: IMKTextInput) -> Character? {
        (textBeforeCursor(client) ?? Self.history).last
    }

    private func cursorLine(_ client: IMKTextInput) -> NSRect {
        var line = NSRect.zero
        client.attributes(forCharacterIndex: 0, lineHeightRectangle: &line)
        return line
    }

    // MARK: IMK

    override func commitComposition(_ sender: Any!) {
        guard !keys.isEmpty, let client = sender as? IMKTextInput else { return }
        commit(keys, client)
    }

    override func deactivateServer(_ sender: Any!) {
        if let client = sender as? IMKTextInput, !keys.isEmpty { commit(keys, client) }
        CandidatePanel.shared.hide()
        super.deactivateServer(sender)
    }

    override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        AppDelegate.active = self
    }

    override func menu() -> NSMenu! {
        let menu = NSMenu()
        let current = CandidatePanel.shared.appearance
        for (i, (title, a)) in [("跟随系统", Appearance.system), ("浅色", .light), ("深色", .dark)].enumerated() {
            let item = NSMenuItem(title: title, action: #selector(pickAppearance(_:)), keyEquivalent: "")
            item.tag = i
            item.state = a == current ? .on : .off
            menu.addItem(item)
        }
        return menu
    }

    /// IMK sends menu actions with a dictionary holding the item.
    @objc func pickAppearance(_ sender: Any?) {
        let item = (sender as? [String: Any])?[kIMKCommandMenuItemName] as? NSMenuItem ?? sender as? NSMenuItem
        guard let tag = item?.tag else { return }
        CandidatePanel.shared.appearance = [Appearance.system, .light, .dark][tag]
    }
}

/// Chinese punctuation for the ASCII keys, while typing Chinese. After a digit, "." and "," stay ASCII (3.14, 1,000).
enum Punctuation {
    private static var openSingle = true, openDouble = true
    private static let map: [String: String] = [
        ",": "，", ".": "。", "?": "？", "!": "！", ";": "；", ":": "：", "\\": "、",
        "(": "（", ")": "）", "[": "【", "]": "】", "<": "《", ">": "》",
        "^": "……", "_": "——", "$": "￥", "~": "～",
    ]

    static func chinese(for s: String, after prev: Character?) -> String? {
        if (s == "." || s == ",") && (prev?.isASCII == true && prev?.isNumber == true) { return nil }
        if s == "'" { defer { openSingle.toggle() }; return openSingle ? "‘" : "’" }
        if s == "\"" { defer { openDouble.toggle() }; return openDouble ? "“" : "”" }
        return map[s]
    }
}
