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
    private var waiting: [NSEvent]?          // keys pressed while waiting for the engine's answer, in order
    private var waitForFull = false          // waiting for the full answer (space), not only the first one
    private var waitTimer: Timer?
    private var noWaiting = false            // replaying after the engine did not answer: commit what there is
    private var picked = false               // the user moved the selection: space commits that, no waiting

    override init!(server: IMKServer!, delegate: Any!, client inputClient: Any!) {
        super.init(server: server, delegate: delegate, client: inputClient)
    }

    var testClient: IMKTextInput?  // shell/tools/keys: IMK only takes its own client proxies
    private var currentClient: IMKTextInput? { testClient ?? client() }

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
        if waiting != nil {
            if Self.replayable(event) {
                waiting!.append(event)
                return true
            }
            stopWaiting(client)  // a key only the app can act on: commit what there is first, keeping the order
        }
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
            if waitForAnswer(event, full: true) { return true }
            commit(model.current ?? keys, client)
            return true
        case 123, 124:  // left / right move the selection, on into the next or previous page
            guard composing else { return false }
            model.move(event.keyCode == 123 ? -1 : 1)
            picked = true
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
                if waitForAnswer(event) { return true }
                if let c = model.slot(n) { commit(c, client) }
                return true
            }
            if composing && (chars == "=" || chars == "-") {  // = / - turn the page
                turnPage(chars == "=" ? 1 : -1, client)
                return true
            }
            if Self.replayable(event) && waitForAnswer(event) { return true }  // punctuation and the like
            if Punctuation.chineseOn, let p = Punctuation.chinese(for: chars, after: composing ? nil : lastChar(client)) {
                if composing { commitBest(client) }
                insert(p, client)
                return true
            }
        }
        if composing { commitBest(client) }
        return false
    }

    // MARK: waiting for the engine's answer (4A.8)

    /// A key that commits while there is no candidate yet (typed fast: "d" then space) waits for the engine's
    /// first answer instead of committing the letters; space also waits for the full answer, up to 0.4 s, unless
    /// the user picked a candidate (D-024: the quick guess's first is often wrong). Keys pressed meanwhile queue.
    private func waitForAnswer(_ event: NSEvent, full: Bool = false) -> Bool {
        guard !keys.isEmpty, !noWaiting, EngineClient.shared.connected else { return false }
        let needFull = full && !picked && (stage == "" || stage == "quick")
        guard model.candidates.isEmpty || needFull else { return false }
        waiting = [event]
        waitForFull = needFull
        waitTimer?.invalidate()
        waitTimer = Timer.scheduledTimer(withTimeInterval: needFull ? 0.4 : 1.0, repeats: false) { [weak self] _ in
            self?.waitTimedOut()
        }
        return true
    }

    /// The full answer is late: commit the quick guess; with no answer at all, give the first one up to 1 s.
    /// No answer by then: the letters, as without the engine.
    private func waitTimedOut() {
        guard let client = currentClient else { return }
        if waitForFull && model.candidates.isEmpty {
            waitForFull = false
            waitTimer = Timer.scheduledTimer(withTimeInterval: 0.6, repeats: false) { [weak self] _ in
                self?.waitTimedOut()
            }
            return
        }
        replay(client, force: true)
    }

    /// Keys that can wait in the queue: whatever ends up as text (letters, digits, punctuation, space).
    /// Others (arrows, delete, return, shortcuts) only the app can act on, after the queue.
    private static func replayable(_ event: NSEvent) -> Bool {
        if !event.modifierFlags.intersection([.command, .control, .option]).isEmpty { return false }
        if event.keyCode == 49 { return true }
        guard let cs = event.characters, cs.unicodeScalars.count == 1, let c = cs.unicodeScalars.first else { return false }
        return c.value >= 0x21 && c.value < 0x7f || c.value >= 0xa0 && !(0xf700...0xf8ff).contains(c.value)
    }

    /// Replays the queued keys in order; one that starts another wait takes the rest with it.
    /// `force`: the first one commits what there is (the letters if no answer came) instead of waiting again.
    private func replay(_ client: IMKTextInput, force: Bool = false) {
        guard var rest = waiting else { return }
        waiting = nil
        waitTimer?.invalidate()
        waitTimer = nil
        var first = force
        while !rest.isEmpty {
            let e = rest.removeFirst()
            noWaiting = first
            first = false
            let handled = handle(e, client: client)
            noWaiting = false
            if !handled, let cs = e.characters { insert(cs, client) }  // what the app would have typed
            if waiting != nil {
                waiting! += rest
                return
            }
        }
    }

    /// Gives up waiting altogether: every queued key goes out now, each with what there is.
    private func stopWaiting(_ client: IMKTextInput) {
        while waiting != nil { replay(client, force: true) }
    }

    /// Shift pressed and released with nothing in between toggles Chinese / English.
    private func flagsChanged(_ event: NSEvent, _ client: IMKTextInput) -> Bool {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if flags == .shift {
            shiftAlone = true
        } else if flags.isEmpty && shiftAlone {
            shiftAlone = false
            stopWaiting(client)  // what was typed before the switch goes out as Chinese
            Self.english.toggle()
            if Self.english && !keys.isEmpty { commit(keys, client) }
        } else {
            shiftAlone = false
        }
        return false
    }

    // MARK: composing

    private func update(_ client: IMKTextInput) {
        // A thin underline only (user: 「不出这个蓝色的」). A plain string, or IMK's raw-text style, has no underline
        // of its own, and each app draws its default: Chromium-based ones a thick underline on a blue background.
        let marked = NSAttributedString(string: keys, attributes: [
            .underlineStyle: NSUnderlineStyle.single.rawValue, .markedClauseSegment: 0])
        client.setMarkedText(marked, selectionRange: NSRange(location: keys.utf16.count, length: 0),
                             replacementRange: NSRange(location: NSNotFound, length: 0))
        wantNextPage = false
        picked = false
        let ctx = context ?? String(Self.history.suffix(64))
        requestId = EngineClient.shared.type(context: ctx, keys: keys)
        if model.candidates.isEmpty { return }  // the panel appears with the first answer
        show(client, fade: false)
    }

    /// The engine answers each key with stage "quick", then "full" (five), then after a pause "more": the full
    /// answer's five followed by further candidates, which only add pages (the page on screen stays as it is).
    func engineAnswered(id: Int, stage: String, candidates: [String]) {
        guard id == requestId, !keys.isEmpty, let client = currentClient else { return }
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
        if waiting != nil && !candidates.isEmpty && (!waitForFull || stage != "quick") {
            replay(client)
            if keys.isEmpty { return }
        }
        show(client, fade: fade)
    }

    /// Turns the page; a next page the engine has not sent yet is turned to when it comes.
    private func turnPage(_ step: Int, _ client: IMKTextInput) {
        if step > 0 && !model.hasNext && stage == "full" {
            wantNextPage = true
            return
        }
        model.turn(step)
        picked = true
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
        picked = false
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
        guard let client = sender as? IMKTextInput else { return }
        stopWaiting(client)
        if !keys.isEmpty { commit(keys, client) }
    }

    override func deactivateServer(_ sender: Any!) {
        if let client = sender as? IMKTextInput {
            stopWaiting(client)
            if !keys.isEmpty { commit(keys, client) }
        }
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
        menu.addItem(.separator())
        for (i, title) in ["标点：英文", "标点：中文"].enumerated() {
            let item = NSMenuItem(title: title, action: #selector(pickPunctuation(_:)), keyEquivalent: "")
            item.tag = i
            item.state = (i == 1) == Punctuation.chineseOn ? .on : .off
            menu.addItem(item)
        }
        return menu
    }

    @objc func pickPunctuation(_ sender: Any?) {
        let item = (sender as? [String: Any])?[kIMKCommandMenuItemName] as? NSMenuItem ?? sender as? NSMenuItem
        guard let tag = item?.tag else { return }
        Punctuation.chineseOn = tag == 1
    }

    /// IMK sends menu actions with a dictionary holding the item.
    @objc func pickAppearance(_ sender: Any?) {
        let item = (sender as? [String: Any])?[kIMKCommandMenuItemName] as? NSMenuItem ?? sender as? NSMenuItem
        guard let tag = item?.tag else { return }
        CandidatePanel.shared.appearance = [Appearance.system, .light, .dark][tag]
    }
}

/// Chinese punctuation for the ASCII keys, while typing Chinese, when chosen in the menu (default: the keys as
/// they are, user 4A.9). After a digit, "." and "," stay ASCII (3.14, 1,000).
enum Punctuation {
    static var chineseOn: Bool {
        get { UserDefaults.standard.string(forKey: "punctuation") == "chinese" }
        set { UserDefaults.standard.set(newValue ? "chinese" : "english", forKey: "punctuation") }
    }
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
