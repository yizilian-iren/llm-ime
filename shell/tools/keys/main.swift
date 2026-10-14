import AppKit
import InputMethodKit

// Checks InputController's key handling against a fake engine (4A.8, 4A.9, 4A.10), and the typing log with its
// summary (5A.2, in a temporary folder):  shell/build.sh keys
// The fake engine answers each key string with a quick guess and a full answer after set delays; the fake app
// keeps the text the input method commits, and types the keys the input method passes on, as an app would.
// Shows the candidate panel on this Mac's screen while it runs (a few seconds).

final class AppDelegate { static weak var active: InputController? }

var failed = 0
func check(_ ok: Bool, _ what: String, line: Int = #line) {
    if !ok { failed += 1; print("FAIL line \(line): \(what)") }
}

// MARK: fake engine

struct Engine {
    static var quick: Double? = 0.05   // seconds to the quick guess; nil: never
    static var full: Double? = 0.25
    static var more: Double? = nil     // the pause's answer, with candidates for the start of the keys (4A.10)
    static var moreExtra: [String] = []  // more whole candidates in that answer, after F: and F2:
    static var contexts: [String: String] = [:]  // keys -> the context they came with (read on the main thread)
}
let sock = NSTemporaryDirectory() + "llmime-keys-\(getpid()).sock"
setenv("LLMIME_SOCK", sock, 1)
let logDir = NSTemporaryDirectory() + "llmime-keys-log-\(getpid())"
setenv("LLMIME_LOG_DIR", logDir, 1)
unlink(sock)
let listener = socket(AF_UNIX, SOCK_STREAM, 0)
var addr = sockaddr_un()
addr.sun_family = sa_family_t(AF_UNIX)
withUnsafeMutableBytes(of: &addr.sun_path) { dst in
    sock.utf8CString.withUnsafeBytes { dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: $0.prefix(dst.count - 1))) }
}
_ = withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
    bind(listener, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) } }
listen(listener, 4)
let writes = DispatchQueue(label: "fake-engine")
Thread.detachNewThread {
    let fd = accept(listener, nil, nil)
    var buf = Data()
    var chunk = [UInt8](repeating: 0, count: 4096)
    func reply(_ id: Int, _ stage: String, _ cands: [String], after: Double?, extra: [String: Any] = [:]) {
        guard let after = after else { return }
        writes.asyncAfter(deadline: .now() + after) {
            let msg = ["id": id, "stage": stage, "candidates": cands].merging(extra) { a, _ in a } as [String: Any]
            var d = try! JSONSerialization.data(withJSONObject: msg)
            d.append(0x0a)
            _ = d.withUnsafeBytes { send(fd, $0.baseAddress!, $0.count, 0) }
        }
    }
    while true {
        let n = recv(fd, &chunk, chunk.count, 0)
        if n <= 0 { break }
        buf.append(chunk, count: n)
        while let nl = buf.firstIndex(of: 0x0a) {
            let line = buf[buf.startIndex..<nl]
            buf = Data(buf[(nl + 1)...])
            guard let m = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                  let id = m["id"] as? Int, let k = m["keys"] as? String else { continue }
            let ctx = m["context"] as? String ?? ""
            DispatchQueue.main.async { Engine.contexts[k] = ctx }
            reply(id, "quick", ["Q:" + k, "Q2:" + k], after: Engine.quick)
            reply(id, "full", ["F:" + k, "F2:" + k], after: Engine.full)
            reply(id, "more", ["F:" + k, "F2:" + k] + Engine.moreExtra, after: Engine.more,
                  extra: ["segments": [["S" + k.prefix(2), 2], ["W" + k, k.count]]])
        }
    }
}

// MARK: fake app

final class FakeClient: NSObject, IMKTextInput {
    var text = ""
    var marked = ""
    func insertText(_ string: Any!, replacementRange: NSRange) { text += "\(string!)"; marked = "" }
    var markedStyle: Any?
    func setMarkedText(_ string: Any!, selectionRange: NSRange, replacementRange: NSRange) {
        let a = string as? NSAttributedString
        marked = a?.string ?? "\(string!)"
        markedStyle = a?.length ?? 0 > 0 ? a?.attribute(.underlineStyle, at: 0, effectiveRange: nil) : nil
    }
    func selectedRange() -> NSRange { NSRange(location: NSNotFound, length: 0) }  // as a terminal: no text to read
    func markedRange() -> NSRange { NSRange(location: NSNotFound, length: 0) }
    func attributedSubstring(from range: NSRange) -> NSAttributedString! { nil }
    func length() -> Int { text.utf16.count }
    func characterIndex(for point: NSPoint, tracking mappingMode: IMKLocationToOffsetMappingMode,
                        inMarkedRange: UnsafeMutablePointer<ObjCBool>!) -> Int { 0 }
    func attributes(forCharacterIndex index: Int, lineHeightRectangle lineRect: UnsafeMutablePointer<NSRect>!) -> [AnyHashable: Any]! {
        lineRect?.pointee = NSRect(x: 200, y: 400, width: 2, height: 18)
        return [:]
    }
    func validAttributesForMarkedText() -> [Any]! { [] }
    func overrideKeyboard(withKeyboardNamed keyboardUniqueName: String!) {}
    func selectMode(_ modeIdentifier: String!) {}
    func supportsUnicode() -> Bool { true }
    func bundleIdentifier() -> String! { "test" }
    func windowLevel() -> CGWindowLevel { 0 }
    func supportsProperty(_ property: TSMDocumentPropertyTag) -> Bool { false }
    func uniqueClientIdentifierString() -> String! { "test" }
    func string(from range: NSRange, actualRange: NSRangePointer!) -> String! { nil }
    func firstRect(forCharacterRange aRange: NSRange, actualRange: NSRangePointer!) -> NSRect { .zero }
}

// MARK: driving it

let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
let client = FakeClient()
let ctl = InputController(server: nil, delegate: nil, client: nil)!
ctl.testClient = client
AppDelegate.active = ctl
EngineClient.shared.onResult = { id, stage, cands, segs in
    ctl.engineAnswered(id: id, stage: stage, candidates: cands, segments: segs)
}
EngineClient.shared.start()

func wait(_ s: Double) { RunLoop.main.run(until: Date(timeIntervalSinceNow: s)) }
let deadline = Date(timeIntervalSinceNow: 5)
while !EngineClient.shared.connected && Date() < deadline { wait(0.05) }
check(EngineClient.shared.connected, "connected to the fake engine")

/// Presses the keys one after another without waiting; what the input method passes on, the app types.
func press(_ keys: [String]) {
    for k in keys {
        let code: UInt16 = ["space": 49, "delete": 51, "right": 124, "down": 125, "return": 36, "esc": 53][k] ?? 0
        let chars = ["space": " ", "delete": "\u{7f}", "right": "\u{f703}", "down": "\u{f701}", "return": "\r",
                     "esc": "\u{1b}"][k] ?? k
        let e = NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0,
                                 context: nil, characters: chars, charactersIgnoringModifiers: chars,
                                 isARepeat: false, keyCode: code)!
        if !ctl.handle(e, client: client) {
            if code == 51 { if !client.text.isEmpty { client.text.removeLast() } } else if code != 36 { client.text += chars }
        }
    }
}
func letters(_ s: String) -> [String] { s.map { String($0) } }
func run(_ name: String, _ keys: [String], settle: Double = 1.5) -> String {
    client.text = ""
    press(keys)
    wait(settle)
    return client.text
}

// Warm up: the first panel takes a while to build, longer than the waits measured below.
var w0 = Date()
press(["w"]); wait(0.5); press(["return"])
print("warm-up \(Int(Date().timeIntervalSince(w0) * 1000)) ms")

// 4A.8, 1: no candidate yet: space waits for the engine (here the full answer, D-024), never "d".
Engine.quick = 0.05; Engine.full = 0.25
check(run("d space", ["d", "space"]) == "F:d", "d + space at once -> the full answer's first: \(client.text)")
// Keys pressed while waiting queue up, in order; punctuation is English by default (4A.9).
var t = run("queue", ["d", "space", "e", "space", ",", "x", "space"])
check(t == "F:dF:e,F:x", "queued keys, in order: \(t)")
// Digits wait for the first answer too.
t = run("digit", ["d", "2"])
check(t == "Q2:d" || t == "F2:d", "d + 2 at once -> the second candidate: \(t)")
// The full answer late: space commits the quick guess after 0.4 s.
Engine.quick = 0.05; Engine.full = 3
let t0 = Date()
client.text = ""
press(["d", "space"])
while client.text.isEmpty && Date().timeIntervalSince(t0) < 2 { wait(0.01) }
let waited = Date().timeIntervalSince(t0)
check(client.text == "Q:d", "full answer late -> the quick guess: \(client.text)")
check(waited > 0.35 && waited < 0.7, "waited ~0.4 s for the full answer: \(waited)")
wait(3)  // let the late answers pass
// Late again for the next word: each space waits for its own full answer, then takes the quick guess.
t = run("late twice", ["d", "space", "e", "space"], settle: 1.2)
check(t == "Q:dQ:e", "two words, full answers late -> both quick guesses: \(t)")
wait(3)
// The user picked a candidate: space commits it at once.
Engine.quick = 0.05; Engine.full = 3
client.text = ""
press(["d"]); wait(0.2); press(["right", "space"])
check(client.text == "Q2:d", "picked with the arrow -> that one, no wait: \(client.text)")
wait(3)
// No answer at all: the letters after ~1 s, as without the engine.
Engine.quick = nil; Engine.full = nil
t = run("silent", ["d", "space", "a"], settle: 1.6)
check(t == "d", "engine silent -> the letters: \(t)")  // "a" starts a new composition
press(["return"])
check(client.text == "da", "then return commits the new letters: \(client.text)")
// A key only the app can act on (delete) ends the wait: what there is goes first, then the app deletes.
Engine.quick = 0.3; Engine.full = 0.6
t = run("delete", ["d", "space", "delete"], settle: 1)
check(t == "", "d space delete -> committed, then deleted by the app: '\(t)'")
// 4A.2: the keys are marked with a thin underline (no app default: a blue background in Chromium).
client.text = ""
press(["d", "a"])
check(client.marked == "da" && client.markedStyle as? Int == NSUnderlineStyle.single.rawValue,
      "marked text with a single underline: \(client.marked) \(String(describing: client.markedStyle))")
press(["return"])
wait(1)

// 4A.10: after a pause, candidates for the start of the keys follow the whole ones; picking one commits only
// it, and the rest of the keys get candidates of their own, after what was just committed.
Engine.quick = 0.05; Engine.full = 0.1; Engine.more = 0.2
client.text = ""
press(letters("abcd")); wait(0.5)
check(client.marked == "abcd", "composing abcd: \(client.marked)")
press(["3"])  // F:abcd, F2:abcd, then Sab (2 keys), Wabcd (all keys)
check(client.text == "Sab" && client.marked == "cd", "picked the start: \(client.text) + marked \(client.marked)")
wait(0.5)
check(Engine.contexts["cd"]?.hasSuffix("Sab") == true, "the rest comes after what was committed: \(Engine.contexts["cd"] ?? "-")")
press(["space"]); wait(0.3)
check(client.text == "SabF:cd" && client.marked.isEmpty, "then the rest: \(client.text)")
client.text = ""
press(letters("abcd")); wait(0.5); press(["4"])
check(client.text == "Wabcd", "one that uses all the keys commits as a whole: \(client.text)")
client.text = ""
press(letters("abcd")); wait(0.5); press(["right", "right", ","])
check(client.text == "F:abcd,", "punctuation with the start selected: the best whole one: \(client.text)")
Engine.more = nil
wait(0.5)

// 4A.9: Chinese punctuation when chosen in the menu.
Engine.quick = 0.05; Engine.full = 0.25
Punctuation.chineseOn = true
t = run("chinese punctuation", ["n", "i", ","])
check(t == "F:ni，" || t == "Q:ni，", "chinese punctuation: \(t)")
Punctuation.chineseOn = false
t = run("english punctuation", ["n", "i", ","])
check(t == "F:ni," || t == "Q:ni,", "english punctuation: \(t)")
UserDefaults.standard.removeObject(forKey: "punctuation")

// 5A.2: the typing log. One record per commit or Esc, written off the key path; the summary counts them.
Engine.quick = 0.05; Engine.full = 0.1; Engine.more = 0.2
Engine.moreExtra = ["F3", "F4", "F5", "F6", "F7"]
press(["esc"])  // writes the last record of the tests above
TypingLog.shared.flush(); TypingLog.shared.deleteAll(); TypingLog.shared.flush()
client.text = ""
press(letters("abcd")); wait(0.5); press(["down"]); press(["2"])  // the next page's second: F7
check(client.text == "F7", "page down, 2 -> the 7th: \(client.text)")
press(letters("abx")); wait(0.3); press(["delete", "c"]); wait(0.3); press(["space"])  // a wrong key, deleted
wait(0.3)
press(["delete", "delete"])  // right after the commit: it was wrong
press(["z", "return"])        // none of them: the letters
press(["q"]); wait(0.2); press(["esc"])
press(["esc"])  // not composing: writes the record kept for deletes after it
TypingLog.shared.flush()
func records() -> [[String: Any]] {
    let files = (try? FileManager.default.contentsOfDirectory(atPath: logDir))?.filter { $0.hasSuffix(".jsonl") } ?? []
    return files.flatMap { f -> [[String: Any]] in
        let s = (try? String(contentsOfFile: logDir + "/" + f, encoding: .utf8)) ?? ""
        return s.split(separator: "\n").compactMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] }
    }
}
var rs = records()
check(rs.map { $0["how"] as? String ?? "" } == ["digit", "space", "enter", "esc"], "records: \(rs.map { $0["how"] ?? "" })")
if rs.count == 4 {
    let (a, b, c, d) = (rs[0], rs[1], rs[2], rs[3])
    check(a["keys"] as? String == "abcd" && a["text"] as? String == "F7" && a["picked"] as? Int == 6
          && a["page"] as? Int == 1 && a["app"] as? String == "test" && (a["cands"] as? [String])?.count == 9,
          "paged pick: \(a)")
    check(((a["events"] as? [[Any]]) ?? []).contains { $0.count == 3 && $0[1] as? String == "page" }, "page event: \(a)")
    let answers = (a["answers"] as? [[Any]]) ?? []
    check(["quick", "full", "more"].allSatisfy { st in answers.contains { $0[1] as? String == st } }, "answer times: \(answers)")
    check(a["t"] is String && a["ms"] is Int && a["context"] is String && a["stage"] as? String == "more", "fields: \(a)")
    let evs = ((b["events"] as? [[Any]]) ?? []).map { $0[1] as? String ?? "" }
    check(b["keys"] as? String == "abc" && evs == ["a", "b", "x", "del", "c"], "deleted and typed again: \(evs)")
    check(b["deleted_after"] as? Int == 2 && b["text"] as? String == "F:abc", "deleted after the commit: \(b)")
    check(c["keys"] as? String == "z" && c["text"] as? String == "z" && c["deleted_after"] as? Int == 0, "enter: \(c)")
    check(d["keys"] as? String == "q" && d["text"] is NSNull, "esc: \(d)")
}
// The summary: one of each, as counted.
let py = Process()
py.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
py.arguments = ["../scripts/typing_summary.py", "--dir", logDir]
py.standardOutput = FileHandle.nullDevice
try? py.run(); py.waitUntilExit()
let mdFile = ((try? FileManager.default.contentsOfDirectory(atPath: logDir)) ?? []).first { $0.hasSuffix(".md") }
let md = mdFile.flatMap { try? String(contentsOfFile: logDir + "/" + $0, encoding: .utf8) } ?? ""
for row in ["| 没选第 1 个 / 翻了页 | 1 |", "| 打了又删、重打 | 1 |", "| 上屏后马上删 | 1 |", "| Enter 上原字母 | 1 |"] {
    check(md.contains(row), "summary has \(row)")
}
check(md.contains("上屏 3 次") && md.contains("放弃（Esc / 删光）1 次"), "summary totals")
// Paused: nothing written.
TypingLog.shared.paused = true
press(letters("pp")); wait(0.2); press(["space", "esc"])
TypingLog.shared.flush()
check(records().count == 4, "paused: no new record (\(records().count))")
TypingLog.shared.paused = false
UserDefaults.standard.removeObject(forKey: "logPaused")
// Deleting everything empties the folder.
TypingLog.shared.deleteAll(); TypingLog.shared.flush()
check(((try? FileManager.default.contentsOfDirectory(atPath: logDir)) ?? ["?"]).isEmpty, "deleted all")
Engine.more = nil; Engine.moreExtra = []
try? FileManager.default.removeItem(atPath: logDir)

unlink(sock)
print(failed == 0 ? "keys: all passed" : "keys: \(failed) failed")
exit(failed == 0 ? 0 : 1)
