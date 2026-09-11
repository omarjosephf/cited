"""Execute the generic demo's inline script against synthetic v2 responses."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).parents[1] / "src" / "assistant" / "static" / "index.html"
NODE = shutil.which("node")

HARNESS = r"""
const fs = require("node:fs");
const vm = require("node:vm");

const scenario = process.argv.at(-1);
const html = fs.readFileSync(process.argv.at(-2), "utf8");
const match = html.match(/<script nonce="__CSP_NONCE__">([\s\S]*?)<\/script>/);
if (!match) throw new Error("inline script not found");

class Element {
  constructor(id = "") {
    this.id = id;
    this.textContent = "";
    this.hidden = true;
    this.disabled = false;
    this.value = "";
    this.dataset = {};
    this.children = [];
    this.listeners = {};
  }
  addEventListener(name, listener) { this.listeners[name] = listener; }
  append(...children) { this.children.push(...children); }
}

const ids = ["form", "question", "submit", "answer", "sources", "flag", "counter"];
const elements = Object.fromEntries(ids.map((id) => [id, new Element(id)]));
const examples = [new Element("example-one"), new Element("example-two")];
examples[0].dataset.q = "Example one?";
examples[1].dataset.q = "Example two?";

global.document = {
  getElementById(id) { return elements[id]; },
  querySelectorAll(selector) {
    if (selector !== ".examples button") throw new Error("unexpected selector");
    return examples;
  },
  createElement(tag) { return new Element(tag); },
};

const timers = [];
global.setTimeout = (callback, milliseconds) => {
  const timer = { callback, milliseconds, cleared: false };
  timers.push(timer);
  return timer;
};
global.clearTimeout = (timer) => { timer.cleared = true; };

const evidence = "a".repeat(64);
const citation = {
  source_id: "guide.md",
  evidence_id: evidence,
  quote: "Quoted fact.",
};
const answered = {
  version: 2,
  state: "answered",
  answer: "A sourced answer.",
  citations: [citation],
};

if (scenario === "max_bounds") {
  answered.answer = "x".repeat(4000);
  answered.citations = Array(8).fill({
    source_id: "s".repeat(200),
    evidence_id: evidence,
    quote: "q".repeat(1000),
  });
}
if (scenario === "invalid_answer") answered.answer = "x".repeat(4001);
if (scenario === "invalid_citations") answered.citations = Array(9).fill(citation);
if (scenario === "invalid_quote") {
  answered.citations = [{ ...citation, quote: "x".repeat(1001) }];
}
if (scenario === "invalid_source") {
  answered.citations = [{ ...citation, source_id: "x".repeat(201) }];
}
if (scenario === "invalid_evidence") {
  answered.citations = [{ ...citation, evidence_id: "A".repeat(64) }];
}

let askCalls = 0;
let resolveAsk;
const askPending = new Promise((resolve) => { resolveAsk = resolve; });

function responseJson(value) {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

global.fetch = async (url, options = {}) => {
  if (url === "/health") {
    return responseJson({ answers_remaining_today: 9, chunks: 4 });
  }
  if (url !== "/ask") throw new Error("unexpected URL");
  askCalls += 1;
  if (scenario === "guard") return askPending;
  if (scenario === "not_covered") {
    return responseJson({ version: 2, state: "not-covered", policy: "unsupported" });
  }
  if (scenario === "not_covered_prose") {
    return responseJson({
      version: 2,
      state: "not-covered",
      policy: "unsupported",
      answer: "untrusted provider prose",
    });
  }
  if (scenario === "unavailable") {
    return responseJson({ version: 2, state: "unavailable" });
  }
  if (scenario === "oversized") return new Response("x".repeat(48 * 1024 + 1));
  if (scenario === "timeout") {
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => reject(new Error("aborted")));
    });
  }
  if (scenario === "body_timeout") {
    return new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode('{"version":2'));
        options.signal.addEventListener(
          "abort", () => controller.error(new Error("aborted"))
        );
      },
    }));
  }
  return responseJson(answered);
};

const tick = () => new Promise((resolve) => setImmediate(resolve));

(async () => {
  vm.runInThisContext(match[1], { filename: "demo-inline.js" });
  await tick();
  elements.question.value = "What is documented?";
  elements.form.listeners.submit({ preventDefault() {} });

  const active = {
    submitDisabled: elements.submit.disabled,
    examplesDisabled: examples.every((button) => button.disabled),
  };

  if (scenario === "guard") {
    elements.form.listeners.submit({ preventDefault() {} });
    examples[0].listeners.click();
    resolveAsk(responseJson(answered));
  }
  if (scenario === "timeout" || scenario === "body_timeout") {
    const timeout = timers.find(
      (timer) => timer.milliseconds === 10000 && !timer.cleared
    );
    if (!timeout) throw new Error("10 second request timer not armed");
    timeout.callback();
  }

  await tick();
  await tick();
  await tick();

  const source = elements.sources.children[1]?.children[1]?.textContent || "";
  process.stdout.write(JSON.stringify({
    askCalls,
    active,
    submitDisabledAfter: elements.submit.disabled,
    examplesDisabledAfter: examples.some((button) => button.disabled),
    answer: elements.answer.textContent,
    flag: elements.flag.textContent,
    source,
    armedTenSecondTimer: timers.some((timer) => timer.milliseconds === 10000),
  }));
})().catch((error) => {
  process.stderr.write(String(error.stack || error));
  process.exitCode = 1;
});
"""


def run_demo(scenario: str) -> dict[str, object]:
    assert NODE is not None, "Node.js is required for the demo script contract test"
    completed = subprocess.run(
        [NODE, "-e", HARNESS, str(PAGE), scenario],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    parsed = json.loads(completed.stdout)
    assert isinstance(parsed, dict)
    return parsed


def test_answered_v2_uses_source_id_and_one_active_request() -> None:
    result = run_demo("guard")

    assert result["askCalls"] == 1
    assert result["active"] == {
        "submitDisabled": True,
        "examplesDisabled": True,
    }
    assert result["submitDisabledAfter"] is False
    assert result["examplesDisabledAfter"] is False
    assert result["answer"] == "A sourced answer."
    assert result["source"] == "guide.md"
    assert result["armedTenSecondTimer"] is True


def test_non_answer_states_use_fixed_generic_copy() -> None:
    not_covered = run_demo("not_covered")
    unavailable = run_demo("unavailable")

    assert not_covered["answer"] == ""
    assert not_covered["flag"] == "I don't have that information in these documents."
    assert unavailable["answer"] == ""
    assert unavailable["flag"] == (
        "The answering service is unavailable. Please try again."
    )


def test_answered_field_limits_are_inclusive() -> None:
    result = run_demo("max_bounds")

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) == 4000
    assert result["source"] == "s" * 200
    assert result["flag"] == ""


def test_not_covered_never_displays_arbitrary_response_prose() -> None:
    result = run_demo("not_covered_prose")

    assert "untrusted provider prose" not in json.dumps(result)
    assert result["flag"] == "The answering service is unavailable. Please try again."


@pytest.mark.parametrize(
    "scenario",
    [
        "invalid_answer",
        "invalid_citations",
        "invalid_quote",
        "invalid_source",
        "invalid_evidence",
        "oversized",
    ],
)
def test_invalid_or_oversized_answered_results_fail_to_fixed_unavailable(
    scenario: str,
) -> None:
    result = run_demo(scenario)

    assert result["answer"] == ""
    assert result["source"] == ""
    assert result["flag"] == "The answering service is unavailable. Please try again."


@pytest.mark.parametrize("scenario", ["timeout", "body_timeout"])
def test_timeout_aborts_fetch_or_body_and_restores_every_submit_path(
    scenario: str,
) -> None:
    result = run_demo(scenario)

    assert result["armedTenSecondTimer"] is True
    assert result["flag"] == "The answering service is unavailable. Please try again."
    assert result["submitDisabledAfter"] is False
    assert result["examplesDisabledAfter"] is False


def test_inline_script_uses_text_only_dom_updates() -> None:
    page = PAGE.read_text(encoding="utf-8")
    script = page.split('<script nonce="__CSP_NONCE__">', 1)[1].split("</script>", 1)[0]

    assert ".innerHTML" not in script
    assert "textContent" in script
    assert "MAX_RESPONSE_BYTES = 48 * 1024" in script
