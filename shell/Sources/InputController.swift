import AppKit
import InputMethodKit

/// One per text client. Letters build the key string (shown as marked text); the engine service answers
/// each change with a quick guess, then the full search, shown in the candidate panel. Whole sentences only.
@objc(LLMInputController)
final class InputController: IMKInputController {
    private static var english = false      // Shift toggles; shared by every client
    private static var history = ""         // what this input method committed, for apps that hide their text

    private var keys = ""
    private var lastLine: NSRect?             // the last cursor line a client gave that is one, in this composition (6B.4)
    private var englishKeys = false          // this composition is in the English mode (6B.2)
    private var spelled: (word: Bool, guesses: [String], completions: [String]) = (true, [], [])
    private var context: String?             // text before the cursor when this composition began
    private var model = CandidateModel()
    private var requestId = 0
    private var stage = ""                   // of the candidates shown; "" after a key: they are for fewer keys (5A.4)
    private var answeredKeys = 0             // how many keys the candidates shown are for
    private var wantNextPage = false         // asked for the next page before the engine's further pages came
    private var shiftAlone = false
    private var waiting: [NSEvent]?          // keys pressed while waiting for the engine's answer, in order
    private var waitForFull = false          // waiting for the full answer (space), not only the first one
    private var waitTimer: Timer?
    private var fullGivenUp = false          // space waited 0.4 s with nothing for these keys: the first answer commits
    private var noWaiting = false            // replaying after the engine did not answer: commit what there is
    private var picked = false               // the user moved the selection: space commits that, no waiting
    private var segmentEnds: [Int] = []      // after the whole candidates: the keys each start-of-keys one uses
    private var comp: Composition?           // this composition, for the typing log (5A.2); nil: not logged
    private var waitStart = Date()
    private var logged: ([String: Any], Date, [(String, String, String)])?  // the last commit's record, written at the
                                             // next key but delete, and the picks to learn from (5A.3) then
    private var pieces: (keys: String, text: String, first: String)?  // start-of-keys picks so far (4A.10)
    private var deletesAfter = 0             // deletes right after that commit (it was wrong)
    private var longest = ""                 // the most keys this composition had (more than now: some deleted, 5A.5)
    private var longestFirst = ""            // the first candidate shown for them
    private var retry: (keys: String, first: String, text: String, used: Int, at: Date)?  // keys given up on, and
                                             // what was committed since (5A.5)
    private var loggedTimer: Timer?
    private var replaying = false            // handle() is replaying a queued key: its own time is not the key's lag (6A.4)
    private var markedMs = 0.0               // what the last update() spent in setMarkedText / in showing (6A.4)
    private var showMs = 0.0
    private var lates: [Int: (shown: String, text: String, at: Date)] = [:]  // quick guesses committed before the full
                                             // answer, by request id: the first they showed, what was committed (6A.4)
    private var held: [Int: ([String: Any], Date)] = [:]  // their records, kept up to 1.5 s for that answer

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
        let started = ProcessInfo.processInfo.systemUptime  // 6A.4: the key's own time to here, and the time in here
        let lag = replaying ? -1 : Self.tenths(started - event.timestamp)
        shiftAlone = false
        if keys.isEmpty && waiting == nil && CandidatePanel.shared.isShown { CandidatePanel.shared.hide() }  // 6A.3
        if event.keyCode != 51 { writeLogged() }
        if event.keyCode == 49, waiting == nil, !keys.isEmpty, comp?.seen == nil {  // what space was pressed on (5A.4)
            comp?.seen = (model.candidates.first, stage, answeredKeys)
        }
        if waiting != nil {
            if Self.replayable(event) {
                waiting!.append(event)
                return true
            }
            stopWaiting(client)  // a key only the app can act on: commit what there is first, keeping the order
        }

        let mods = event.modifierFlags.intersection([.command, .control, .option])
        if !mods.isEmpty {
            if !keys.isEmpty { commit(keys, client, how: "shortcut") }  // a shortcut: leave the app nothing half-typed
            return false
        }
        let chars = event.characters ?? ""
        let composing = !keys.isEmpty

        switch event.keyCode {
        case 51:  // delete
            guard composing else {
                if logged != nil { deletesAfter += 1 }
                return false
            }
            keys.removeLast()
            comp?.event("del")
            if keys.isEmpty {
                gaveUp(longest, first: longestFirst)
                log(how: "deleted", text: nil)
                reset(client)
            } else {
                update(client, keyed: (started, lag))
            }
            return true
        case 53:  // esc
            guard composing else { return false }
            gaveUp(longest, first: longestFirst)
            log(how: "esc", text: nil)
            reset(client)
            return true
        case 36, 76:  // return, enter: the letters as typed
            guard composing else { return false }
            commit(keys, client, how: "enter")
            return true
        case 49:  // space
            guard composing else { return false }
            if englishKeys {  // the word and the space after it, as typing English without candidates
                commitSelected(model.selected, client, how: "space")
                insert(" ", client)
                return true
            }
            if waitForAnswer(event, full: true) { return true }
            commitSelected(model.selected, client, how: "space")
            return true
        case 123, 124:  // left / right move the selection, on into the next or previous page
            guard composing else { return false }
            model.move(event.keyCode == 123 ? -1 : 1)
            comp?.event("move", event.keyCode == 123 ? -1 : 1)
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
            if c.isASCII && (c.properties.isLowercase || Self.english && c.properties.isUppercase) {
                if !composing && Self.english {  // the spell checker needs no engine; nothing logged or learned
                    englishKeys = true
                    context = textBeforeCursor(client)
                }
                if !composing && !Self.english && !EngineClient.shared.connected { return false }  // no engine: letters
                if !composing && !Self.english {
                    let read = ProcessInfo.processInfo.systemUptime
                    context = textBeforeCursor(client)
                    let readMs = Self.tenths(ProcessInfo.processInfo.systemUptime - read)
                    let app = client.bundleIdentifier()
                    comp = TypingLog.shared.logs(app: app) ? Composition(app: app) : nil
                    comp?.ctxMs = readMs
                }
                keys.append(Character(c))
                comp?.event(String(c))
                update(client, keyed: (started, lag))
                return true
            }
            if composing, let n = Int(chars), (1...CandidateModel.pageSize).contains(n) {
                if waitForAnswer(event) { return true }
                if model.slot(n) != nil { commitSelected(model.pageRange.lowerBound + n - 1, client, how: "digit") }
                return true
            }
            if composing && !englishKeys && (chars == "=" || chars == "-") {  // = / - turn the page
                turnPage(chars == "=" ? 1 : -1, client)
                return true
            }
            if Self.replayable(event) && waitForAnswer(event) { return true }  // punctuation and the like
            if Punctuation.chineseOn, !Self.english, let p = Punctuation.chinese(for: chars, after: composing ? nil : lastChar(client)) {
                if composing { commitBest(client, how: "punct") }
                insert(p, client)
                return true
            }
        }
        if composing { commitBest(client, how: Self.replayable(event) ? "punct" : "other") }
        return false
    }

    // MARK: waiting for the engine's answer (4A.8)

    /// A key that commits while there is no candidate yet (typed fast: "d" then space) waits for the engine's
    /// first answer instead of committing the letters; space also waits for the full answer, up to 0.4 s, unless
    /// the user picked a candidate (D-024: the quick guess's first is often wrong). Keys pressed meanwhile queue.
    /// Candidates still for fewer keys (typed fast: the last key's answer not there yet) count as none (5A.4).
    private func waitForAnswer(_ event: NSEvent, full: Bool = false) -> Bool {
        guard !keys.isEmpty, !noWaiting, EngineClient.shared.connected else { return false }
        let needFull = full && !picked && (stage == "" || stage == "quick")
        guard !hasAnswer || needFull else { return false }
        waiting = [event]
        waitStart = Date()
        waitForFull = needFull
        if needFull { EngineClient.shared.hurry() }
        fullGivenUp = false
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
        if waitForFull && !hasAnswer {
            waitForFull = false
            fullGivenUp = true
            waitTimer = Timer.scheduledTimer(withTimeInterval: 0.6, repeats: false) { [weak self] _ in
                self?.waitTimedOut()
            }
            return
        }
        replay(client, force: true)
    }

    /// Candidates for the keys as they are now: an answer for fewer keys is not one.
    private var hasAnswer: Bool { !model.candidates.isEmpty && stage != "" }

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
        fullGivenUp = false
        comp?.event("wait", Int(Date().timeIntervalSince(waitStart) * 1000))
        waitTimer?.invalidate()
        waitTimer = nil
        var first = force
        while !rest.isEmpty {
            let e = rest.removeFirst()
            noWaiting = first
            first = false
            replaying = true
            let handled = handle(e, client: client)
            replaying = false
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

    /// Shift pressed and released with nothing in between toggles Chinese / English. In English the letters show
    /// candidates too (6B.2): the word as typed, corrections, completions, the English of pinyin; Enter types them
    /// as they are, space the selected one and a space, punctuation the selected one and itself.
    private func flagsChanged(_ event: NSEvent, _ client: IMKTextInput) -> Bool {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if flags == .shift {
            shiftAlone = true
        } else if flags.isEmpty && shiftAlone {
            shiftAlone = false
            stopWaiting(client)  // what was typed before the switch goes out as Chinese
            if !keys.isEmpty { commit(keys, client, how: "switch") }  // either way: the keys as typed
            Self.english.toggle()
        } else {
            shiftAlone = false
        }
        return false
    }

    // MARK: composing

    /// Milliseconds with one decimal.
    private static func tenths(_ seconds: TimeInterval) -> Double { (seconds * 10_000).rounded() / 10 }

    /// A key that changed the keys: update(), and what it cost the main thread (6A.4).
    private func update(_ client: IMKTextInput, keyed: (at: TimeInterval, lag: Double)) {
        update(client)
        comp?.shell.append([keys.count, keyed.lag, Self.tenths(ProcessInfo.processInfo.systemUptime - keyed.at),
                            markedMs, showMs])
    }

    private func update(_ client: IMKTextInput) {
        // A thin underline only (user: 「不出这个蓝色的」): a plain string, or IMK's raw-text style, has no underline of
        // its own, and apps drew their default (a thick line, on blue in Chromium-based ones). The line's color is the
        // app's, from the system Accent color: our underlineColor does not get through (on M3 the line followed the
        // Accent color), and no attribute makes Chromium draw none, so the shell does not draw its own (D-027).
        let marked = NSAttributedString(string: keys, attributes: [
            .underlineStyle: NSUnderlineStyle.single.rawValue, .underlineColor: CandidatePanel.shared.accent,
            .markedClauseSegment: 0])
        let marking = ProcessInfo.processInfo.systemUptime
        client.setMarkedText(marked, selectionRange: NSRange(location: keys.utf16.count, length: 0),
                             replacementRange: NSRange(location: NSNotFound, length: 0))
        markedMs = Self.tenths(ProcessInfo.processInfo.systemUptime - marking)
        showMs = 0
        wantNextPage = false
        picked = false
        if englishKeys {
            spelled = English.spelling(keys)
            model.candidates = English.candidates(keys, word: spelled.word, guesses: spelled.guesses,
                                                  completions: spelled.completions)
            model.selected = 0
            stage = "english"
            answeredKeys = keys.count
            if EngineClient.shared.connected {
                requestId = EngineClient.shared.type(context: context ?? Self.history, keys: keys, english: true)
            }
            show(client, fade: false)
            return
        }
        if keys.count > longest.count { (longest, longestFirst) = (keys, "") }
        let ctx = context ?? String(Self.history.suffix(64))
        requestId = EngineClient.shared.type(context: ctx, keys: keys)
        comp?.context = ctx
        comp?.appText = context != nil
        comp?.sent[requestId] = (keys.count, Date())
        stage = ""  // the candidates shown are for the keys before this one, until its answer comes (5A.4)
        if model.candidates.isEmpty { return }  // the panel appears with the first answer
        let showing = ProcessInfo.processInfo.systemUptime
        show(client, fade: false)
        showMs = Self.tenths(ProcessInfo.processInfo.systemUptime - showing)
    }

    /// The engine answers each key with stage "quick", then "full" (five), then after a pause "more": the full
    /// answer's five followed by further candidates, which only add pages (the page on screen stays as it is),
    /// and after those the candidates for the start of the keys (4A.10), so the pages do not run out.
    func engineAnswered(id: Int, stage: String, candidates: [String], segments: [(String, Int)] = [], by: String? = nil) {
        if stage == "full" { lateFull(id, candidates) }
        guard id == requestId, !keys.isEmpty, let client = currentClient else { return }
        let started = ProcessInfo.processInfo.systemUptime
        let n = keys.count
        defer { comp?.busy.append([n, stage, Self.tenths(ProcessInfo.processInfo.systemUptime - started)]) }  // 6A.4
        if stage == "english" {  // the English of the keys read as pinyin, merged into the spelling's (6B.2)
            guard englishKeys, !candidates.isEmpty else { return }
            model.candidates = English.candidates(keys, word: spelled.word, guesses: spelled.guesses,
                                                  completions: spelled.completions, pinyin: candidates)
            model.selected = 0
            show(client, fade: false)
            return
        }
        comp?.answered(id: id, stage: stage, by: by)
        answeredKeys = keys.count
        if stage == "more" {
            guard self.stage == "full" else { return }
            self.stage = stage
            model.candidates = candidates + segments.map { $0.0 }
            segmentEnds = segments.map { $0.1 }
            if wantNextPage {
                wantNextPage = false
                model.turn(1)
            }
            show(client, fade: false)
            return
        }
        let fade = stage == "full" && self.stage == "quick" && !model.candidates.isEmpty
        comp?.showing(candidates, keys: keys.count, stage: stage)
        self.stage = stage
        model.candidates = candidates
        segmentEnds = []
        model.selected = 0
        if keys == longest, let f = candidates.first { longestFirst = f }
        if waiting != nil && !candidates.isEmpty && (!waitForFull || stage != "quick") {
            replay(client, force: fullGivenUp)
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
        comp?.event("page", step)
        picked = true
        show(client, fade: false)
    }

    /// Only while keys are being typed here (6A.3); the panel asks `wanted` again while it is up.
    private func show(_ client: IMKTextInput, fade: Bool) {
        guard !keys.isEmpty else { return CandidatePanel.shared.hide() }
        CandidatePanel.shared.show(model, below: cursorLine(client), fade: fade) { [weak self] in
            !(self?.keys.isEmpty ?? true)
        }
    }

    /// Before a key that is not ours: the selection, but the best whole candidate if that is only the start.
    /// Never one for fewer keys (a wait given up): the letters then, as with no answer.
    private func commitBest(_ client: IMKTextInput, how: String) {
        let i = segmentEnd(model.selected) == nil ? model.selected : 0
        guard hasAnswer, model.candidates.indices.contains(i) else { return commit(keys, client, how: "letters") }
        commit(model.candidates[i], client, how: how, picked: i)
    }

    /// The keys candidate `i` uses when it is one for the start of the keys only (nil: the whole keys).
    private func segmentEnd(_ i: Int) -> Int? {
        let j = i - (model.candidates.count - segmentEnds.count)
        guard segmentEnds.indices.contains(j), segmentEnds[j] < keys.count else { return nil }
        return segmentEnds[j]
    }

    /// Commits candidate `i`. One for the start of the keys commits only itself (4A.10): the rest of the keys stay
    /// marked and get candidates of their own, after the context plus what was just committed.
    private func commitSelected(_ i: Int, _ client: IMKTextInput, how: String) {
        guard hasAnswer, model.candidates.indices.contains(i) else { return commit(keys, client, how: "letters") }
        let text = model.candidates[i]
        guard let end = segmentEnd(i) else { return commit(text, client, how: how, picked: i) }
        insert(text, client)
        EngineClient.shared.commit(text)
        log(how: how, picked: i, text: text, used: end)
        let p = pieces ?? ("", "", model.candidates.first ?? "")
        pieces = (p.keys + keys.prefix(end), p.text + text, p.first)
        if let c = comp { comp = Composition(app: c.app) }  // the rest of the keys: a record of its own
        keys = String(keys.dropFirst(end))
        longest = String(longest.dropFirst(end))
        longestFirst = ""
        if context != nil { context! += text }
        model = CandidateModel()
        segmentEnds = []
        stage = ""
        answeredKeys = 0
        CandidatePanel.shared.hide()  // back with the first answer for the rest
        update(client)
    }

    private func commit(_ text: String, _ client: IMKTextInput, how: String, picked: Int? = nil) {
        var learns: [(String, String, String)] = []
        if picked != nil, let first = model.candidates.first {  // a candidate, not the letters
            let p = pieces ?? ("", "", first)
            if longest.count > keys.count { gaveUp(p.keys + longest, first: p.text + longestFirst) }  // typed more, deleted back
            learns.append((p.keys + keys, p.text + text, p.first))
            learns += retried(p.keys + keys, p.text + text)
        } else {
            retry = nil
        }
        pieces = nil
        if picked != nil, stage == "quick", requestId != 0, let first = model.candidates.first {  // the full answer may
            let id = requestId                                                                  // still come (6A.4)
            lates[id] = (first, text, Date())
            Timer.scheduledTimer(withTimeInterval: 1.5, repeats: false) { [weak self] _ in self?.lateFull(id, nil) }
        }
        log(how: how, picked: picked, text: text, learns: learns)
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
        lastLine = nil
        englishKeys = false
        longest = ""
        longestFirst = ""
        context = nil
        comp = nil
        pieces = nil
        model = CandidateModel()
        segmentEnds = []
        requestId = 0
        stage = ""
        answeredKeys = 0
        wantNextPage = false
        picked = false
        CandidatePanel.shared.hide()
    }

    // MARK: the typing log (5A.2)

    /// Keeps this composition's record until the next key, which tells whether it was deleted right away; a pick
    /// is learned from (5A.3) only if it was not. Paused or an excluded app: neither.
    private func log(how: String, picked: Int? = nil, text: String?, used: Int? = nil,
                     learns: [(String, String, String)] = []) {
        guard let comp = comp else { return }
        writeLogged()
        var r = comp.record(keys: keys, model: model, stage: stage, how: how, picked: picked, text: text, used: used)
        if lates[requestId] != nil { r["late_id"] = requestId }
        logged = (r, Date(), learns)
        loggedTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: false) { [weak self] _ in self?.writeLogged() }
    }

    private func writeLogged() {
        loggedTimer?.invalidate()
        loggedTimer = nil
        guard let (r, date, learns) = logged else { return }
        var record = r
        record["deleted_after"] = deletesAfter
        if deletesAfter == 0 {
            for (k, t, f) in learns { EngineClient.shared.learn(keys: k, text: t, first: f) }
        } else if !learns.isEmpty, let k = r["keys"] as? String, let t = r["text"] as? String {
            let kept = String(t.dropLast(deletesAfter))  // what was left of it stays, its keys taken as typed again
            gaveUp(k, first: t, text: kept, used: k.count * kept.count / t.count)
        }
        logged = nil
        deletesAfter = 0
        if let id = record["late_id"] as? Int, lates[id] != nil {
            held[id] = (record, date)
            return
        }
        record["late_id"] = nil
        TypingLog.shared.write(record, at: date)
    }

    /// The full answer to a quick guess that was committed (6A.4), or none when it did not come in time: how late it
    /// came, whether its first is the one that was shown / committed, whether the committed one is among its first
    /// five. Counts only, no text. Writes the record that waited for it.
    private func lateFull(_ id: Int, _ candidates: [String]?) {
        guard let l = lates.removeValue(forKey: id) else { return }
        var v: [Any] = []  // none: it did not come
        if let c = candidates {
            v = [Int(Date().timeIntervalSince(l.at) * 1000), c.first == l.shown ? 1 : 0, c.first == l.text ? 1 : 0,
                 c.prefix(5).contains(l.text) ? 1 : 0]
        }
        if let (held, date) = held.removeValue(forKey: id) {
            var r = held
            r["late_id"] = nil
            r["late_full"] = v
            TypingLog.shared.write(r, at: date)
        } else if (logged?.0["late_id"] as? Int) == id {
            logged!.0["late_full"] = v
        }
    }

    // MARK: given up and typed again (5A.5)

    /// The candidates for `keys` were given up on (Esc, deleted, typed more and deleted back, a commit deleted right
    /// after): what is committed next, until as many keys were typed again, is what the user meant (`text`: of a
    /// commit, what was left of it). Within 20 s of each other; anything but a candidate ends it.
    private func gaveUp(_ keys: String, first: String, text: String = "", used: Int = 0) {
        retry = keys.count >= 2 ? (keys, first, text, used, Date()) : nil
    }

    /// A commit after keys were given up on: with what was committed since, learned for those keys. The engine keeps
    /// it only where it reads them, so one character at a time is learned once the last one is in (鸣 then 潮 for
    /// mingchao: 鸣潮). The first commit alone is a pick like any, learned as one already.
    private func retried(_ keys: String, _ text: String) -> [(String, String, String)] {
        guard var r = retry, Date().timeIntervalSince(r.at) < 20 else { retry = nil; return [] }
        let alone = r.text.isEmpty
        r.text += text
        r.used += keys.count
        r.at = Date()
        retry = r.used < r.keys.count ? r : nil
        return alone ? [] : [(r.keys, r.text, r.first)]
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

    /// The cursor line the candidates sit under. Index 0 of the marked text, as most clients answer it; a client that
    /// answers with a rectangle that is no line on any screen (zero, a whole text field, off the screen: some Qt and
    /// web views) is asked again at the marked and the selected range, and failing that the last good line of this
    /// composition is kept, else the pointer (6B.4).
    private func cursorLine(_ client: IMKTextInput) -> NSRect {
        var indices = [0]
        for r in [client.markedRange(), client.selectedRange()] where r.location != NSNotFound && !indices.contains(r.location) {
            indices.append(r.location)
        }
        for i in indices {
            var line = NSRect.zero
            client.attributes(forCharacterIndex: i, lineHeightRectangle: &line)
            if Self.isLine(line) {
                lastLine = line
                return line
            }
        }
        if let last = lastLine { return last }
        let p = NSEvent.mouseLocation
        return NSRect(x: p.x, y: p.y, width: 1, height: 16)
    }

    /// A rectangle a text line can be: on a screen, not zero, not taller than a few lines.
    private static func isLine(_ r: NSRect) -> Bool {
        r.height >= 4 && r.height <= 120 && r.width < 2000 && !(r.origin == .zero)
            && NSScreen.screens.contains { $0.frame.contains(NSPoint(x: r.minX, y: r.midY)) }
    }

    // MARK: IMK

    override func commitComposition(_ sender: Any!) {
        guard let client = sender as? IMKTextInput else { return }
        stopWaiting(client)
        if !keys.isEmpty { commit(keys, client, how: "away") }
    }

    override func deactivateServer(_ sender: Any!) {
        if let client = sender as? IMKTextInput {
            stopWaiting(client)
            if !keys.isEmpty { commit(keys, client, how: "away") }
        }
        writeLogged()
        CandidatePanel.shared.hide()
        super.deactivateServer(sender)
    }

    override func hidePalettes() {
        CandidatePanel.shared.hide()
        super.hidePalettes()
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
        menu.addItem(.separator())
        let pause = NSMenuItem(title: "暂停记录", action: #selector(togglePause(_:)), keyEquivalent: "")
        pause.state = TypingLog.shared.paused ? .on : .off
        menu.addItem(pause)
        // 6A.2: scripts/retrain.py on M4 reads it over ssh (defaults read com.iren.inputmethod.LLMIME retrainPaused)
        let retrain = NSMenuItem(title: "暂停每周重训", action: #selector(toggleRetrain(_:)), keyEquivalent: "")
        retrain.state = UserDefaults.standard.bool(forKey: "retrainPaused") ? .on : .off
        menu.addItem(retrain)
        let remote = NSMenuItem(title: "远程计算", action: #selector(toggleRemote(_:)), keyEquivalent: "")  // 7E.1
        remote.state = EngineClient.remoteOn ? .on : .off
        menu.addItem(remote)
        let passes = NSMenuItem(title: "两遍计算（先快后准）", action: #selector(togglePasses(_:)), keyEquivalent: "")  // 7D.2
        passes.state = EngineClient.twoPasses ? .on : .off
        menu.addItem(passes)
        menu.addItem(NSMenuItem(title: "打开记录文件夹", action: #selector(openLog(_:)), keyEquivalent: ""))
        menu.addItem(NSMenuItem(title: "删除全部记录", action: #selector(deleteLog(_:)), keyEquivalent: ""))
        return menu
    }

    @objc func togglePause(_ sender: Any?) { TypingLog.shared.paused.toggle() }
    @objc func toggleRetrain(_ sender: Any?) {
        UserDefaults.standard.set(!UserDefaults.standard.bool(forKey: "retrainPaused"), forKey: "retrainPaused")
    }
    @objc func toggleRemote(_ sender: Any?) { EngineClient.remoteOn.toggle() }
    @objc func togglePasses(_ sender: Any?) { EngineClient.twoPasses.toggle() }
    @objc func openLog(_ sender: Any?) { TypingLog.shared.openFolder() }
    @objc func deleteLog(_ sender: Any?) {
        logged = nil  // the last commit's record too
        held = [:]
        TypingLog.shared.deleteAll()
        EngineClient.shared.forget()  // and the word list learned from them (5A.3)
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
