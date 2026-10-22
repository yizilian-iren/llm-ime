import AppKit

/// The local typing log (5A.2, D-019, D-030): one JSON line per commit or Esc, in a file per day under
/// ~/Library/Application Support/llm-ime/log/ (only on this Mac). scripts/typing_summary.py reads a day and
/// writes where typing was hard beside it. The menu pauses it, opens the folder, deletes everything.
/// Lines are encoded and written on the log's own queue, never on the key path.
final class TypingLog {
    static let shared = TypingLog()

    static var directory: URL {
        if let p = ProcessInfo.processInfo.environment["LLMIME_LOG_DIR"] { return URL(fileURLWithPath: p) }  // tests
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("llm-ime/log")
    }

    /// Password managers: never logged. More with `defaults write com.iren.inputmethod.LLMIME logExclude -array id ...`
    static let excluded: Set<String> = [
        "com.1password.1password", "com.agilebits.onepassword7", "com.agilebits.onepassword-osx",
        "com.apple.keychainaccess", "com.apple.Passwords", "com.bitwarden.desktop", "com.lastpass.LastPass",
        "org.keepassxc.keepassxc", "in.sinew.Enpass-Desktop", "com.dashlane.dashlanephonefinal",
    ]

    var paused: Bool {
        get { UserDefaults.standard.bool(forKey: "logPaused") }
        set { UserDefaults.standard.set(newValue, forKey: "logPaused") }
    }

    func logs(app: String?) -> Bool {
        guard !paused else { return false }
        guard let app = app else { return true }
        return !Self.excluded.contains(app) && !(UserDefaults.standard.stringArray(forKey: "logExclude") ?? []).contains(app)
    }

    private let queue = DispatchQueue(label: "llm-ime.log", qos: .utility)
    private let day: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()
    private let stamp: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        f.timeZone = .current
        return f
    }()

    func write(_ record: [String: Any], at date: Date) {
        queue.async {
            var r = record
            r["t"] = self.stamp.string(from: date)
            guard var data = try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys, .withoutEscapingSlashes])
            else { return }
            data.append(0x0a)
            let dir = Self.directory
            let url = dir.appendingPathComponent(self.day.string(from: date) + ".jsonl")
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true,
                                                     attributes: [.posixPermissions: 0o700])
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
            }
            guard let h = try? FileHandle(forWritingTo: url) else { return }
            defer { try? h.close() }
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: data)
        }
    }

    /// Waits for the lines written so far (tests).
    func flush() { queue.sync {} }

    /// Every file in the log folder: the days' records and their summaries.
    func deleteAll() {
        queue.async {
            let dir = Self.directory
            for f in (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [] {
                try? FileManager.default.removeItem(at: f)
            }
        }
    }

    func openFolder() {
        try? FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true,
                                                 attributes: [.posixPermissions: 0o700])
        NSWorkspace.shared.open(Self.directory)
    }
}

/// What happened in one composition, up to its commit: the keys as pressed (with ms since its first key), the
/// engine's answer times, waits. InputController keeps one while composing and hands it to TypingLog.
struct Composition {
    let start = Date()
    let app: String?
    var context = ""                       // what the engine got as the text before the cursor
    var appText = false                    // read from the app (else: what this input method committed lately)
    var events: [[Any]] = []               // [ms, "a"] letter, [ms, "del"], [ms, "page", ±1], [ms, "move", ±1],
                                           // [ms, "wait", ms waited for an answer before committing (4A.8)]
    var answers: [[Any]] = []              // [keys count, "quick" / "full" / "more", ms from the key to the answer]
    var sent: [Int: (Int, Date)] = [:]     // request id -> (keys count, when sent)
    var seen: (String?, String, Int)?      // at the first space: the first candidate on screen, its stage ("": for
                                           // fewer keys, 5A.4) and how many keys it is for
    var shell: [[Any]] = []                // per letter / delete key (6A.4): [keys count after it, ms from the key's own
                                           // time to handle() (-1: replayed after a wait), ms in handle(), of it in
                                           // setMarkedText, of it in showing the candidates]
    var busy: [[Any]] = []                 // per answer shown (6A.4): [keys count, stage, ms engineAnswered took]
    var flips: [[Any]] = []                // each time the first five on screen changed (6A.4): [ms, keys count, stage,
                                           // 1: for the same keys as the ones before it (quick -> full), 1: the first changed]
    var ctxMs: Double?                     // the first key: ms reading the text before the cursor from the app (6A.4)
    private var shown: [String] = []       // the first five on screen, and how many keys they were for
    private var shownKeys = 0

    init(app: String?) { self.app = app }

    var ms: Int { Int(Date().timeIntervalSince(start) * 1000) }

    mutating func event(_ e: Any...) { events.append([ms] + e) }

    /// The candidates an answer put on screen (6A.4): counts a change of the first five, and of the first.
    mutating func showing(_ candidates: [String], keys n: Int, stage: String) {
        let top = Array(candidates.prefix(5))
        defer { shown = top; shownKeys = n }
        guard !shown.isEmpty, shown != top else { return }
        flips.append([ms, n, stage, n == shownKeys ? 1 : 0, shown.first == top.first ? 0 : 1])
    }

    mutating func answered(id: Int, stage: String) {
        guard let (n, at) = sent[id] else { return }
        answers.append([n, stage, Int(Date().timeIntervalSince(at) * 1000)])
        if stage == "more" { sent[id] = nil }
    }

    /// The record of its commit. `how`: space, digit, segment (4A.10: only the start of the keys), enter (the
    /// letters as typed), punct / other (a key the app acts on, after the best one), letters (no candidates),
    /// shortcut / switch / away (committed as typed because of a shortcut, Shift to English, leaving the app),
    /// esc (given up: nothing committed).
    func record(keys: String, model: CandidateModel, stage: String, how: String, picked: Int?, text: String?,
                used: Int? = nil) -> [String: Any] {
        var r: [String: Any] = [
            "app": app ?? NSNull(), "context": context, "app_text": appText, "keys": keys, "events": events,
            "answers": answers, "cands": Array(model.candidates.prefix(40)), "stage": stage, "how": how,
            "picked": picked ?? NSNull(), "page": model.page, "text": text ?? NSNull(), "ms": ms,
        ]
        if let used = used { r["used"] = used }
        if !shell.isEmpty { r["shell"] = shell }
        if !busy.isEmpty { r["busy"] = busy }
        if !flips.isEmpty { r["flips"] = flips }
        if let c = ctxMs { r["ctx_ms"] = c }
        if let (first, stage, n) = seen {
            let s: [String: Any] = ["first": first ?? NSNull(), "stage": stage, "keys": n]
            r["seen"] = s
        }
        return r
    }
}
