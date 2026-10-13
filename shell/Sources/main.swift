import AppKit
import InputMethodKit

/// The input method's process: an IMK server (the connection name is in Info.plist) and the engine client.
final class AppDelegate: NSObject, NSApplicationDelegate {
    static weak var active: InputController?
    var server: IMKServer?

    func applicationDidFinishLaunching(_ note: Notification) {
        let name = Bundle.main.infoDictionary?["InputMethodConnectionName"] as? String ?? "LLMIME_Connection"
        server = IMKServer(name: name, bundleIdentifier: Bundle.main.bundleIdentifier)
        EngineClient.shared.onResult = { id, stage, cands in
            AppDelegate.active?.engineAnswered(id: id, stage: stage, candidates: cands)
        }
        EngineClient.shared.start()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
