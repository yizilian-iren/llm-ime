import AppKit

/// The English mode's candidates (6B.2, D-025): the keys as typed, macOS's spelling corrections and completions for
/// them, and from the engine the English of the keys read as pinyin (pingguo: apple). A real word comes first as
/// typed; a word that is not one, its first correction, then the keys as typed (user: 「可以，实际使用我再调整」);
/// keys that are not a word but are whole pinyin, their English first.
enum English {
    static let pageMax = 20

    /// The spell checker's view of the keys: a word as typed, its corrections, its completions.
    static func spelling(_ keys: String) -> (word: Bool, guesses: [String], completions: [String]) {
        let sc = NSSpellChecker.shared
        let range = NSRange(location: 0, length: keys.utf16.count)
        let wrong = sc.checkSpelling(of: keys, startingAt: 0, language: "en", wrap: false,
                                     inSpellDocumentWithTag: 0, wordCount: nil)
        let word = wrong.location == NSNotFound
        let guesses = word ? [] : sc.guesses(forWordRange: range, in: keys, language: "en", inSpellDocumentWithTag: 0) ?? []
        let completions = sc.completions(forPartialWordRange: range, in: keys, language: "en", inSpellDocumentWithTag: 0) ?? []
        return (word, guesses, completions)
    }

    /// The candidates in order; `pinyin`: the engine's English for the keys read as pinyin (none yet, or not pinyin).
    static func candidates(_ keys: String, word: Bool, guesses: [String], completions: [String],
                           pinyin: [String] = []) -> [String] {
        var order: [String]
        if word {
            order = [keys] + completions.prefix(4) + pinyin + completions.dropFirst(4)
        } else if !pinyin.isEmpty {
            order = pinyin + [keys] + guesses + completions
        } else {
            order = guesses.prefix(1) + [keys] + guesses.dropFirst() + completions
        }
        var seen = Set<String>()
        return Array(order.filter { seen.insert($0).inserted }.prefix(pageMax))
    }
}
