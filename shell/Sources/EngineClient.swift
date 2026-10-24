import Foundation

/// The connection to the engine service (ime/server.py): one JSON object per line over a Unix socket.
/// Never blocks the caller: connecting, writing and reading run on their own queue and thread, and while
/// the service is down it retries every 2 s (the shell then types letters straight into the app).
final class EngineClient {
    static let shared = EngineClient()

    static var socketPath: String {
        if let p = ProcessInfo.processInfo.environment["LLMIME_SOCK"] { return p }  // tests (shell/build.sh keys)
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("llm-ime/engine.sock").path
    }

    /// Called on the main thread with (request id, stage "quick" / "full" / "more", candidates, segments, by): with
    /// "more", candidates for the start of the keys and how many keys each uses (4A.10); otherwise none. by: where the
    /// engine computed it, "remote" (home4070) / "local", or nil without remote.json (7E.1).
    var onResult: ((Int, String, [String], [(String, Int)], String?) -> Void)?
    /// Called on the main thread each time the engine connects (or reconnects after a restart): the keys typed while it
    /// was away are sent again (7A.1).
    var onConnect: (() -> Void)?
    private(set) var connected = false  // read on the main thread

    private let queue = DispatchQueue(label: "llm-ime.engine")
    private var fd: Int32 = -1           // owned by `queue`
    private var nextId = 0

    func start() {
        queue.async { self.connect() }
    }

    /// Sends the keys typed so far; returns the request id the answers will carry. `english`: the English mode's
    /// keys, answered once with stage "english": the English of the keys read as pinyin (6B.2).
    @discardableResult
    func type(context: String?, keys: String, english: Bool = false) -> Int {
        nextId += 1
        var msg: [String: Any] = ["id": nextId, "keys": keys]
        msg["context"] = context ?? NSNull()
        if english { msg["mode"] = "english" }
        send(msg)
        return nextId
    }

    func commit(_ text: String) {
        send(["op": "commit", "text": text])
    }

    /// A pick to learn from (5A.3): the keys, what was committed for them, the engine's first candidate then.
    func learn(keys: String, text: String, first: String) {
        send(["op": "learn", "keys": keys, "text": text, "first": first])
    }

    /// Space waits for the full answer to the latest keys: the engine starts it now, not after its pause (5B.12).
    func hurry() {
        send(["op": "hurry"])
    }

    /// The menu's 远程计算 (7E.1): off, the engine computes everything on this Mac. Sent again on each connect.
    static var remoteOn: Bool {
        get { !UserDefaults.standard.bool(forKey: "remoteOff") }
        set { UserDefaults.standard.set(!newValue, forKey: "remoteOff"); shared.send(["op": "remote", "on": newValue]) }
    }

    /// The menu's 两遍计算（先快后准） (7D.2): default off, one pass (quick only). Sent again on each connect.
    static var twoPasses: Bool {
        get { UserDefaults.standard.bool(forKey: "twoPasses") }
        set { UserDefaults.standard.set(newValue, forKey: "twoPasses"); shared.send(["op": "passes", "two": newValue]) }
    }

    /// Clears the engine's word list (删除全部记录).
    func forget() {
        send(["op": "forget"])
    }

    private func send(_ msg: [String: Any]) {
        guard var data = try? JSONSerialization.data(withJSONObject: msg) else { return }
        data.append(0x0a)
        queue.async {
            guard self.fd >= 0 else { return }
            let ok = data.withUnsafeBytes { buf -> Bool in
                var off = 0
                while off < buf.count {
                    let n = Darwin.send(self.fd, buf.baseAddress! + off, buf.count - off, 0)
                    if n <= 0 { return false }
                    off += n
                }
                return true
            }
            if !ok { self.disconnect() }
        }
    }

    private func connect() {
        let s = socket(AF_UNIX, SOCK_STREAM, 0)
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let path = Self.socketPath.utf8CString
        withUnsafeMutableBytes(of: &addr.sun_path) { dst in
            path.withUnsafeBytes { src in dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src.prefix(dst.count - 1))) }
        }
        var nosig: Int32 = 1
        setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &nosig, socklen_t(MemoryLayout<Int32>.size))
        let r = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(s, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        if r != 0 {
            close(s)
            queue.asyncAfter(deadline: .now() + 2) { self.connect() }
            return
        }
        fd = s
        if !Self.remoteOn { send(["op": "remote", "on": false]) }  // queued behind this: fd is set
        send(["op": "passes", "two": Self.twoPasses])
        DispatchQueue.main.async { self.connected = true; self.onConnect?() }
        Thread.detachNewThread { self.read(s) }
    }

    private func disconnect() {
        guard fd >= 0 else { return }
        close(fd)
        fd = -1
        DispatchQueue.main.async { self.connected = false }
        queue.asyncAfter(deadline: .now() + 2) { self.connect() }
    }

    private func read(_ s: Int32) {
        var buf = Data()
        var chunk = [UInt8](repeating: 0, count: 65536)
        while true {
            let n = recv(s, &chunk, chunk.count, 0)
            if n <= 0 { break }
            buf.append(chunk, count: n)
            while let nl = buf.firstIndex(of: 0x0a) {
                let line = buf[buf.startIndex..<nl]
                buf = Data(buf[(nl + 1)...])
                guard let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                      let id = obj["id"] as? Int, let stage = obj["stage"] as? String,
                      let cands = obj["candidates"] as? [String] else { continue }
                let segs = (obj["segments"] as? [[Any]] ?? []).compactMap { s -> (String, Int)? in
                    guard s.count == 2, let t = s[0] as? String, let n = s[1] as? Int else { return nil }
                    return (t, n)
                }
                let by = obj["by"] as? String
                DispatchQueue.main.async { self.onResult?(id, stage, cands, segs, by) }
            }
        }
        queue.async { if self.fd == s { self.disconnect() } }
    }
}
