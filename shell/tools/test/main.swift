import AppKit

// Checks CandidateModel's paging:  shell/build.sh test
var failed = 0
func check(_ ok: Bool, _ what: String, line: Int = #line) {
    if !ok { failed += 1; print("FAIL line \(line): \(what)") }
}

var m = CandidateModel(candidates: (1...12).map { "c\($0)" }, selected: 0)
check(m.pageRange == 0..<5 && !m.hasPrev && m.hasNext, "page 1 of 3")
check(m.slot(3) == "c3" && m.slot(6) == nil, "slots on page 1")
m.turn(1)
check(m.selected == 5 && m.pageRange == 5..<10 && m.hasPrev && m.hasNext, "= to page 2")
check(m.slot(1) == "c6", "slot 1 on page 2")
m.turn(1)
check(m.selected == 10 && m.pageRange == 10..<12 && !m.hasNext, "page 3 has 2")
check(m.slot(3) == nil, "no slot 3 on the last page")
m.turn(1)
check(m.selected == 10, "no page after the last")
m.move(-1)
check(m.selected == 9 && m.page == 1, "left from a page's first goes to the previous page's last")
m.move(1); m.move(1); m.move(1)
check(m.selected == 11, "right stops at the last candidate")
m.turn(-1); m.turn(-1); m.turn(-1)
check(m.selected == 0 && m.current == "c1", "- back to page 1 and no further")

var few = CandidateModel(candidates: ["a", "b", "c"], selected: 0)
check(!few.hasNext && !few.hasPrev, "one page: no arrows")
few.turn(1)
check(few.selected == 0, "no turning with one page")
check(CandidateModel().current == nil, "empty has no current")

let v = CandidateView(frame: .zero)
v.model = CandidateModel(candidates: (1...12).map { "候选\($0)" }, selected: 0)
let paged = v.fittingSize().width
v.model = CandidateModel(candidates: (1...5).map { "候选\($0)" }, selected: 0)
check(paged == v.fittingSize().width + CandidateView.arrowsWidth, "arrows add their width only when paged")

print(failed == 0 ? "paging: all passed" : "paging: \(failed) failed")
exit(failed == 0 ? 0 : 1)
