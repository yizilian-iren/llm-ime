import AppKit
import InputMethodKit

// Checks InputController's key handling against a fake engine (4A.8, 4A.9):  shell/build.sh keys
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
}
let sock = NSTemporaryDirectory() + "llmime-keys-\(getpid()).sock"
setenv("LLMIME_SOCK", sock, 1)
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
    func reply(_ id: Int, _ stage: String, _ cands: [String], after: Double?) {
        guard let after = after else { return }
        writes.asyncAfter(deadline: .now() + after) {
            var d = try! JSONSerialization.data(withJSONObject: ["id": id, "stage": stage, "candidates": cands] as [String: Any])
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
            reply(id, "quick", ["Q:" + k, "Q2:" + k], after: Engine.quick)
            reply(id, "full", ["F:" + k, "F2:" + k], after: Engine.full)
        }
    }
}

// MARK: fake app

final class FakeClient: NSObject, IMKTextInput {
    var text = ""
    var marked = ""
    func insertText(_ string: Any!, replacementRange: NSRange) { text += "\(string!)"; marked = "" }
    func setMarkedText(_ string: Any!, selectionRange: NSRange, replacementRange: NSRange) { marked = "\(string!)" }
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
EngineClient.shared.onResult = { id, stage, cands in ctl.engineAnswered(id: id, stage: stage, candidates: cands) }
EngineClient.shared.start()

func wait(_ s: Double) { RunLoop.main.run(until: Date(timeIntervalSinceNow: s)) }
let deadline = Date(timeIntervalSinceNow: 5)
while !EngineClient.shared.connected && Date() < deadline { wait(0.05) }
check(EngineClient.shared.connected, "connected to the fake engine")

/// Presses the keys one after another without waiting; what the input method passes on, the app types.
func press(_ keys: [String]) {
    for k in keys {
        let code: UInt16 = ["space": 49, "delete": 51, "right": 124, "return": 36][k] ?? 0
        let chars = ["space": " ", "delete": "\u{7f}", "right": "\u{f703}", "return": "\r"][k] ?? k
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
// 4A.9: Chinese punctuation when chosen in the menu.
Engine.quick = 0.05; Engine.full = 0.25
Punctuation.chineseOn = true
t = run("chinese punctuation", ["n", "i", ","])
check(t == "F:ni，" || t == "Q:ni，", "chinese punctuation: \(t)")
Punctuation.chineseOn = false
t = run("english punctuation", ["n", "i", ","])
check(t == "F:ni," || t == "Q:ni,", "english punctuation: \(t)")
UserDefaults.standard.removeObject(forKey: "punctuation")

unlink(sock)
print(failed == 0 ? "keys: all passed" : "keys: \(failed) failed")
exit(failed == 0 ? 0 : 1)
