"""100k quality suite: recall question selection, scoring, tool-call parsing, drift compare."""

import math

from gpubench.quality import (
    Thresholds,
    _kl,
    compare_responses,
    parse_tool_call,
    recall_candidates,
    score_recall,
)


def _session(n=10):
    msgs = [{"role": "user", "content": "fix the bug"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": {
            "name": "run_command", "arguments": {"command": f"grep -n thing_{i} src/"}}}]})
        out = f"src/module_{i}.py:12: def thing_{i}(x): return x\n(Open file: n/a)"
        msgs.append({"role": "tool", "content": out})
    return msgs


def test_recall_candidates_are_deep_unique_and_exact():
    msgs = _session()
    text = "".join(m.get("content", "") for m in msgs)
    cands = recall_candidates(msgs, text)
    assert cands, "expected usable questions"
    for c in cands:
        assert 0.2 <= c["depth"] <= 0.8
        assert c["expected"].startswith("src/module_") and text.count(c["expected"]) == 1
    # A command run twice is ambiguous and never asked about.
    dup = _session()
    dup[3]["tool_calls"][0]["function"]["arguments"]["command"] = "grep -n thing_5 src/"
    assert all(c["cmd"] != "grep -n thing_5 src/" for c in recall_candidates(dup, text))


def test_score_recall():
    assert score_recall("src/a.py:12: def f(x)", "src/a.py:12:  def f(x)\n") == (True, True)
    assert score_recall("src/a.py:12: def f(x)", "It was: src/a.py:12: def f(x)") == (True, False)
    assert score_recall("src/a.py:12", "no idea") == (False, False)


def test_parse_tool_call():
    text = ("Let me look.\n<tool_call>\n<function=run_command>\n<parameter=command>\n"
            "open src/a.py\n</parameter>\n</function>\n</tool_call>")
    assert parse_tool_call(text) == ("run_command", {"command": "open src/a.py"})
    assert parse_tool_call("no call") is None


def test_kl_zero_for_identical_and_positive_otherwise():
    p = {"a": math.log(0.7), "b": math.log(0.3)}
    assert _kl(p, p) == 0
    assert _kl(p, {"a": math.log(0.5), "b": math.log(0.5)}) > 0


def _resp(i, text, tokens, top):
    return {"id": f"drift-{i}", "kind": "drift", "text": text, "tokens": tokens,
            "top_logprobs": top}


CALL = "<function=run_command>\n<parameter=command>\nls\n</parameter>\n</function>"


def test_compare_passes_identical_and_fails_changed_calls():
    top = [{"x": -0.01, "y": -5.0}] * 3
    base = [_resp(i, CALL, ["x", "x", "x"], top) for i in range(10)]
    assert compare_responses(base, base)["pass"]
    other = CALL.replace("ls", "rm -rf /")
    cand = [_resp(i, other if i < 2 else CALL, ["x", "x", "x"], top) for i in range(10)]
    report = compare_responses(base, cand)
    assert report["same_tool_call"] == 0.8 and not report["checks"]["same_tool_call"]
    assert not report["pass"]


def test_compare_flags_distribution_drift_and_recall_drop():
    base = [_resp(0, CALL, ["x", "y"], [{"x": -0.01, "y": -5.0}, {"y": -0.01, "x": -5.0}]),
            {"id": "recall-0", "kind": "recall", "text": "line A", "expected": "line A"}]
    cand = [_resp(0, CALL, ["x", "y"], [{"x": -0.9, "y": -0.5}, {"y": -0.01, "x": -5.0}]),
            {"id": "recall-0", "kind": "recall", "text": "wrong", "expected": "line A"}]
    report = compare_responses(base, cand, Thresholds())
    assert report["mean_kl"] > 0.02 and not report["checks"]["mean_kl"]
    assert report["recall_baseline"] == 1.0 and report["recall_candidate"] == 0.0
    assert not report["checks"]["recall"]
