(function () {
  "use strict";

  // Config lives in a JSON <script> tag, not templated directly, so
  // Jinja's {{ }} never has to parse as valid JS.
  let cfg;
  try {
    cfg = JSON.parse(document.getElementById("dashboard-config").textContent);
  } catch {
    cfg = { ptySupported: false, stageChoices: [], runModeChoices: [] };
  }

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", "false"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("is-active"));
      tab.setAttribute("aria-selected", "true");
      document.getElementById(tab.dataset.panel).classList.add("is-active");
    });
  });

  let toastTimer = null;
  function showToast(message) {
    const root = document.getElementById("toast-root");
    root.innerHTML = "";
    const el = document.createElement("div");
    el.className = "toast";
    el.textContent = message;
    root.appendChild(el);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.remove(), 6000);
  }

  function refreshTables() {
    document.body.dispatchEvent(new CustomEvent("runs-changed"));
  }

  async function postAction(url) {
    let res;
    try {
      res = await fetch(url, { method: "POST" });
    } catch (err) {
      showToast(`Network error: ${err.message}`);
      return;
    }
    if (res.status === 204) {
      refreshTables();
      return;
    }
    let detail = `Action failed (HTTP ${res.status})`;
    try {
      const data = await res.json();
      if (data.detail) detail = data.detail;
    } catch { /* non-JSON error body, keep the generic message */ }
    showToast(detail);
  }

  const ACTION_URLS = {
    trash: (id) => `/api/runs/${encodeURIComponent(id)}/trash`,
    restore: (id) => `/api/trash/${encodeURIComponent(id)}/restore`,
    purge: (id) => `/api/trash/${encodeURIComponent(id)}/purge`,
  };

  // Fires the same CustomEvent postAction() does, on any push from
  // RunsBroadcaster (dashboard.py).
  function connectRunsSocket() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const sock = new WebSocket(`${proto}//${location.host}/ws/runs`);

    sock.addEventListener("message", (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.type === "runs-changed") refreshTables();
    });

    // Reconnect on drop instead of leaving the tab on stale data.
    sock.addEventListener("close", () => {
      setTimeout(connectRunsSocket, 2000);
    });
    sock.addEventListener("error", () => {
      sock.close();
    });
  }
  connectRunsSocket();

  document.body.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-act]");
    if (!btn) return;
    const act = btn.dataset.act;

    // Its own typed-confirmation modal, not window.confirm().
    if (act === "purge-all") {
      openPurgeAllConfirm(parseInt(btn.dataset.runCount, 10) || 0);
      return;
    }
    if (btn.dataset.confirm && !window.confirm(btn.dataset.confirm)) return;
    closeDropdown();

    const runId = btn.dataset.runId;
    const build = ACTION_URLS[act];
    if (build) postAction(build(runId));
  });

  const modalStack = [];

  function openBackdropModal(el) {
    const idx = modalStack.indexOf(el);
    if (idx !== -1) modalStack.splice(idx, 1);
    modalStack.push(el);
    el.classList.add("is-open");
  }
  function closeBackdropModal(el) {
    const idx = modalStack.indexOf(el);
    if (idx !== -1) modalStack.splice(idx, 1);
    el.classList.remove("is-open");
    if (el._resetFields) el._resetFields();
  }

  // Irreversible: stays disabled until the exact run count is typed back in.
  const purgeAllModal = document.getElementById("modal-purge-all");
  const purgeAllPrompt = document.getElementById("purge-all-prompt");
  const purgeAllInput = document.getElementById("purge-all-input");
  const purgeAllConfirmBtn = document.getElementById("purge-all-confirm");
  let purgeAllExpected = "";

  function openPurgeAllConfirm(count) {
    purgeAllExpected = String(count);
    purgeAllPrompt.textContent =
      `This permanently deletes all ${count} trashed run${count === 1 ? "" : "s"} — there is no undo. ` +
      `Type ${purgeAllExpected} to confirm.`;
    purgeAllInput.value = "";
    purgeAllConfirmBtn.disabled = true;
    openBackdropModal(purgeAllModal);
    purgeAllInput.focus();
  }
  function closePurgeAllConfirm() {
    closeBackdropModal(purgeAllModal);
  }
  purgeAllInput.addEventListener("input", () => {
    purgeAllConfirmBtn.disabled = purgeAllInput.value.trim() !== purgeAllExpected;
  });
  purgeAllInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !purgeAllConfirmBtn.disabled) purgeAllConfirmBtn.click();
  });
  purgeAllConfirmBtn.addEventListener("click", () => {
    if (purgeAllInput.value.trim() !== purgeAllExpected) return; // belt & suspenders
    closePurgeAllConfirm();
    postAction("/api/trash/purge-all");
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") return;
    const top = modalStack[modalStack.length - 1];
    if (top) closeBackdropModal(top);
  });
  document.getElementById("purge-all-cancel").addEventListener("click", closePurgeAllConfirm);
  purgeAllModal.addEventListener("click", (ev) => {
    if (ev.target === purgeAllModal) closePurgeAllConfirm();
  });

  const drawer = document.getElementById("term-drawer");
  const statusPill = document.getElementById("term-status-pill");
  const statusText = document.getElementById("term-status-text");
  const argvLabel = document.getElementById("term-argv");
  const xtermContainer = document.getElementById("xterm-container");
  const resizeHandle = document.getElementById("term-resize-handle");
  const consoleToggle = document.getElementById("console-toggle");
  const consoleToggleIcon = document.getElementById("console-toggle-icon");

  let term = null;
  let fitAddon = null;
  let socket = null;
  let sessionLive = false;
  let hasLaunched = false; // false = console toggle should open on an empty terminal
  let currentLaunch = null; // last openDrawer() payload — read by the "exit" handler below

  // Independent of the xterm viewport — "Clear" wipes the screen, not the
  // downloadable log.
  let logChunks = [];

  // Standard strip-ansi pattern, so the downloaded .txt has no raw \x1b[...m codes.
  const ANSI_ESCAPE_RE = /\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g;

  const DRAWER_HEIGHT_KEY = "ug_dashboard.termDrawerHeight";
  const MIN_DRAWER_HEIGHT = 160;

  // Coalesces bursts of resize events into one fit() per animation frame.
  let fitPending = false;
  function scheduleFit() {
    if (fitPending) return;
    fitPending = true;
    requestAnimationFrame(() => {
      fitPending = false;
      if (drawer.classList.contains("is-open")) safeFit();
    });
  }

  function ensureTerm() {
    if (term) return;
    term = new Terminal({
      convertEol: true,
      fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
      fontSize: 13,
      theme: {
        background: "#12161f",
        foreground: "#e4e7ee",
        cursor: "#4dd8e8",
        selectionBackground: "#232937",
        black: "#0b0e14",
        red: "#e8544d",
        green: "#4de8a0",
        yellow: "#e8a23d",
        blue: "#4dd8e8",
        magenta: "#e8544d",
        cyan: "#4dd8e8",
        white: "#e4e7ee",
        brightBlack: "#7c8496",
      },
    });
    fitAddon = new FitAddon.FitAddon();
    term.loadAddon(fitAddon);
    term.open(xtermContainer);
    term.onData((data) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "stdin", data }));
      }
    });

    term.attachCustomKeyEventHandler((ev) => {
      if (ev.type !== "keydown" || !ev.ctrlKey || ev.shiftKey || ev.metaKey || ev.altKey) return true;
      if (ev.key !== "c" && ev.key !== "C") return true;
      if (term.hasSelection()) return true;
      showToast("Ctrl-C is disabled — use Stop above to end the run.");
      return false;
    });

    window.addEventListener("resize", scheduleFit);

    // Catches box changes fitWhenSettled() below doesn't (e.g. an async
    // font load shifting character metrics).
    new ResizeObserver(scheduleFit).observe(xtermContainer);
  }

  function clampDrawerHeight(px) {
    const max = Math.round(window.innerHeight * 0.9);
    return Math.min(Math.max(px, MIN_DRAWER_HEIGHT), max);
  }

  function applyStoredDrawerHeight() {
    const stored = parseInt(localStorage.getItem(DRAWER_HEIGHT_KEY) || "", 10);
    if (!Number.isNaN(stored)) {
      drawer.style.height = clampDrawerHeight(stored) + "px";
    }
  }

  (function setupDrawerResize() {
    let dragging = false;
    let startY = 0;
    let startHeight = 0;

    resizeHandle.addEventListener("pointerdown", (ev) => {
      dragging = true;
      startY = ev.clientY;
      startHeight = drawer.getBoundingClientRect().height;
      drawer.classList.add("is-resizing");
      // Pointer capture so a fast drag doesn't drop the gesture.
      resizeHandle.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    });

    resizeHandle.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const next = clampDrawerHeight(startHeight + (startY - ev.clientY));
      drawer.style.height = `${next}px`;
      safeFit();
    });

    function endDrag(ev) {
      if (!dragging) return;
      dragging = false;
      drawer.classList.remove("is-resizing");
      localStorage.setItem(DRAWER_HEIGHT_KEY, String(Math.round(drawer.getBoundingClientRect().height)));
      try { resizeHandle.releasePointerCapture(ev.pointerId); } catch { /* already released */ }
    }
    resizeHandle.addEventListener("pointerup", endDrag);
    resizeHandle.addEventListener("pointercancel", endDrag);

    resizeHandle.addEventListener("dblclick", () => {
      drawer.style.height = "";
      localStorage.removeItem(DRAWER_HEIGHT_KEY);
      safeFit();
    });
  })();

  document.getElementById("term-clear").addEventListener("click", () => {
    if (term) term.clear();
  });

  document.getElementById("term-download").addEventListener("click", () => {
    if (!logChunks.length) {
      showToast("Nothing to download yet — start a session first.");
      return;
    }
    const text = logChunks.join("").replace(ANSI_ESCAPE_RE, "");
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const a = document.createElement("a");
    a.href = url;
    a.download = `dashboard-run-log-${stamp}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });

  function safeFit() {
    if (!fitAddon || !term) return;
    try {
      fitAddon.fit();
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "resize", rows: term.rows, cols: term.cols }));
      }
    } catch { /* container not visible yet — next open() will retry */ }
  }

  // Waits for transitionend instead of guessing rAF frames — under load
  // (a fresh process's stdout burst) a frame-count guess fires early and
  // undersizes the terminal. Timeout covers transitions too short to
  // fire transitionend (reduced motion).
  function fitWhenSettled() {
    safeFit();

    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      drawer.removeEventListener("transitionend", onEnd);
      clearTimeout(fallback);
      if (drawer.classList.contains("is-open")) safeFit();
    };
    const onEnd = (ev) => {
      if (ev.target === drawer && ev.propertyName === "height") finish();
    };
    const fallback = setTimeout(finish, 260); // transition is 180ms; margin for a slow frame
    drawer.addEventListener("transitionend", onEnd);
  }

  function setStatus(kind, text) {
    statusPill.classList.remove("status-pill--idle", "status-pill--running", "status-pill--done", "status-pill--error");
    statusPill.classList.add(`status-pill--${kind}`);
    statusText.textContent = text;
  }

  function setConsoleToggleUI(isOpen) {
    consoleToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
    consoleToggleIcon.innerHTML = isOpen ? "&#9660;" : "&#9650;";
  }

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Same technique as MiroFish's Step4Report.vue (sequential regex passes,
  // no markdown library) — escaped first, which theirs isn't. Renders one
  // saved search record's body (search.py's report + "## Sources" appendix)
  // for the report panel — see showSearchRecord() further down.
  function renderMarkdown(text) {
    // render_citation_appendix() writes one raw `<a id="cite-E#"></a>` per
    // source heading — the only HTML this function ever has to trust,
    // since it comes from search.py's own code, never the LLM. Pulled out
    // before escaping and spliced back in verbatim at the end, or it would
    // come out as literal "&lt;a...&gt;" text like everything else here.
    const anchors = [];
    const safe = text.replace(/<a id="(cite-E\d+)"><\/a>/g, (_m, id) => {
      anchors.push(id);
      return `\u0000A${anchors.length - 1}\u0000`;
    });

    let html = escapeHtml(safe);
    html = html.replace(/```([\s\S]*?)```/g, "<pre class=\"report-code\"><code>$1</code></pre>");
    html = html.replace(/`([^`]+)`/g, "<code class=\"report-inline-code\">$1</code>");
    html = html.replace(/^#### (.+)$/gm, "<h5>$1</h5>");
    html = html.replace(/^### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^## (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^# (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // Citations: linkify_citations() emits exactly "[\[E12\]](#cite-E12)"
    // (backslash-escaped inner brackets so the nested "[" doesn't end the
    // link label early) — handled before the generic link rule below,
    // which can't parse that escaping.
    html = html.replace(/\[\\\[([^\]]*?)\\\]\]\(#(cite-E\d+)\)/g,
      (_m, label, id) => `<a href="#${id}" class="report-link report-link--cite" data-panel-jump>[${label}]</a>`);
    // Any other Markdown link: an in-page anchor jumps within this same
    // panel (see the click handler on #search-report-body); an http(s)
    // URL opens in a new tab; anything else is left as plain text.
    html = html.replace(/\[([^\]\n]+)\]\((#[\w-]+|https?:\/\/[^\s)]+)\)/g, (_m, label, href) => {
      if (/^https?:\/\//.test(href)) {
        return `<a href="${href}" class="report-link" target="_blank" rel="noopener">${label}</a>`;
      }
      return `<a href="${href}" class="report-link" data-panel-jump>${label}</a>`;
    });
    // _post_line() indents each cited post as "  > line" under its bullet —
    // allow leading whitespace so those still read as quotes instead of
    // silently falling through to plain paragraph text.
    html = html.replace(/^[ \t]*&gt; ?(.*)$/gm, "<blockquote>$1</blockquote>");
    html = html.replace(/^---$/gm, "<hr>");
    html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);
    html = html.split(/\n{2,}/).map((block) => {
      const t = block.trim();
      if (!t) return "";
      return /^<(h\d|ul|blockquote|pre|hr)/.test(t) ? t : `<p>${t}</p>`;
    }).join("\n");

    return html.replace(/\u0000A(\d+)\u0000/g, (_m, i) => `<a id="${anchors[Number(i)]}"></a>`);
  }

  // ─── SEARCH REPORT PANEL ───────────────────────────────────────────────
  // Renders one saved <run_dir>/searches/*.md record (runs_api.py's
  // get_search_record_parsed(), already linkified by search.py) into the
  // wide slide-over declared in dashboard.html (#search-report-backdrop).
  // Reached two ways, both converging on showSearchRecord() so a replayed
  // question looks identical to a fresh one:
  //   - a just-finished CLI/terminal search ("exit" handler below calls
  //     resolveSearchOutcome())
  //   - picking a past question from a run's history dropdown
  //     (_runs_table.html's data-open-search-report, wired further down)
  const searchReportBackdrop = document.getElementById("search-report-backdrop");
  const searchReportQuestion = document.getElementById("search-report-question");
  const searchReportMeta = document.getElementById("search-report-meta");
  const searchReportBody = document.getElementById("search-report-body");

  function showSearchRecord(record) {
    searchReportQuestion.textContent = record.question || "(untitled question)";
    searchReportMeta.innerHTML = "";
    if (record.asked_at) {
      const rt = document.createElement("relative-time");
      rt.setAttribute("datetime", record.asked_at);
      rt.textContent = record.asked_at;
      searchReportMeta.appendChild(rt);
    }
    searchReportBody.innerHTML = renderMarkdown(record.body || "");
    // The scrollable element is .detail-panel__body (dashboard.css), not
    // #search-report-body itself — resetting the wrong one left the panel
    // opening wherever the previous report had been scrolled to.
    const scrollWrap = searchReportBody.closest(".detail-panel__body");
    if (scrollWrap) scrollWrap.scrollTop = 0;
    closeDropdown();
    searchReportBackdrop.classList.add("is-open");
  }

  // Opened the instant a question is submitted, before search.py has
  // written anything — same panel, same chrome, just a progress stepper
  // instead of a report. Step boundaries mirror run_search()'s own
  // "[N/5] ..." / "AI ANALYSIS" log lines one-for-one (search.py), read
  // live off the terminal socket's stdout stream in updateSearchProgress()
  // below, so this tracks the real pipeline rather than a fixed timer.
  const SEARCH_PROGRESS_STEPS = [
    { label: "Decomposing query" },   // search.py [1/5]-[2/5]
    { label: "Pulling content" },     // search.py [3/5]-[5/5]: vector search, merge, component pull
    { label: "Awaiting response" },   // search.py "AI ANALYSIS" — the LLM call itself
  ];

  function showSearchProgress(query) {
    searchReportQuestion.textContent = query;
    searchReportMeta.innerHTML = "";
    searchReportBody.innerHTML =
      '<div class="search-progress">' +
      SEARCH_PROGRESS_STEPS.map((s) =>
        `<div class="search-progress__step"><span class="search-progress__icon"></span>` +
        `<span class="search-progress__label">${escapeHtml(s.label)}</span></div>`
      ).join("") +
      "</div>";
    setSearchProgressStep(0);
    const scrollWrap = searchReportBody.closest(".detail-panel__body");
    if (scrollWrap) scrollWrap.scrollTop = 0;
    closeDropdown();
    searchReportBackdrop.classList.add("is-open");
  }

  function setSearchProgressStep(activeIndex) {
    searchReportBody.querySelectorAll(".search-progress__step").forEach((el, i) => {
      el.classList.toggle("is-done", i < activeIndex);
      el.classList.toggle("is-active", i === activeIndex);
    });
  }

  // Called on every stdout chunk while a search is in flight (see the
  // socket "stdout" handler in openDrawer below). No-ops once the panel
  // has moved on to a real report — querySelector just finds nothing.
  function updateSearchProgressFromLog() {
    if (!searchReportBody.querySelector(".search-progress")) return;
    const text = logChunks.join("");
    let idx = 0;
    if (text.includes("[3/5]") || text.includes("[4/5]") || text.includes("[5/5]")) idx = 1;
    if (text.includes("AI ANALYSIS")) idx = 2;
    setSearchProgressStep(idx);
  }

  function closeSearchReport() {
    searchReportBackdrop.classList.remove("is-open");
  }
  document.getElementById("search-report-close").addEventListener("click", closeSearchReport);
  searchReportBackdrop.addEventListener("click", (ev) => {
    if (ev.target === searchReportBackdrop) closeSearchReport();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && searchReportBackdrop.classList.contains("is-open")) closeSearchReport();
  });

  // Fetches one record by filename and opens it — the single path both the
  // history dropdown and the "just finished" flow below go through.
  async function openSearchReport(runId, filename) {
    let res;
    try {
      res = await fetch(`/api/runs/${encodeURIComponent(runId)}/searches/${encodeURIComponent(filename)}`);
    } catch (err) {
      showToast(`Network error: ${err.message}`);
      return;
    }
    if (!res.ok) {
      let detail = `Couldn't load that search (HTTP ${res.status})`;
      try { const data = await res.json(); if (data.detail) detail = data.detail; } catch { /* non-JSON body */ }
      showToast(detail);
      return;
    }
    showSearchRecord(await res.json());
  }

  // Ordered (newest-first, per list_searches() in runs_api.py) filenames
  // for one run's saved search records.
  async function listSearchFilenames(runId) {
    let html;
    try {
      const res = await fetch(`/partials/runs/${encodeURIComponent(runId)}/searches`);
      if (!res.ok) return [];
      html = await res.text();
    } catch {
      return [];
    }
    const doc = new DOMParser().parseFromString(html, "text/html");
    return Array.from(doc.querySelectorAll("[data-open-search-report]")).map((el) => el.dataset.filename);
  }

  // Search runs show their progress in the report panel instead of the
  // terminal drawer (see openDrawer). On any failure — non-zero exit, a
  // dropped connection, or a zero exit that wrote no new record (search.py
  // swallowed an exception, e.g. an Ollama call failing) — switch back to
  // the drawer so the real process output is visible, instead of leaving
  // the progress screen stuck or opening an unrelated past record.
  function failSearchAndShowLogs(message) {
    closeSearchReport();
    drawer.classList.add("is-open");
    setConsoleToggleUI(true);
    fitWhenSettled();
    if (message) showToast(message);
  }

  // Called once a search process exits with code 0. Compares against the
  // filenames captured at launch time (currentLaunch.knownSearchFiles) so
  // a run that reports success but produced nothing new is treated as a
  // failure rather than silently reopening an older question's answer.
  async function resolveSearchOutcome(launch) {
    const [before, after] = await Promise.all([launch.knownSearchFiles, listSearchFilenames(launch.run_id)]);
    const beforeSet = new Set(before);
    const newFile = after.find((f) => !beforeSet.has(f));
    if (newFile) {
      openSearchReport(launch.run_id, newFile);
    } else {
      failSearchAndShowLogs("Search finished without producing a report — see logs.");
    }
  }

  // Fetches a run's past questions into its history dropdown — see the
  // .split-btn__toggle wiring in _runs_table.html and the generic
  // [data-dropdown-toggle] handler further down, which calls this whenever
  // the panel being opened carries [data-search-history]. Re-fetched on
  // every open, not cached, so a search that just finished shows up
  // immediately without a full table refresh.
  function loadSearchHistory(panel, toggle) {
    panel.innerHTML = '<div class="dropdown__empty">Loading…</div>';
    const runId = panel.dataset.runId;
    fetch(`/partials/runs/${encodeURIComponent(runId)}/searches`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.text();
      })
      .then((html) => {
        panel.innerHTML = html;
        // Loading placeholder had a different height than the real list —
        // repositions against the toggle now that it's settled, but only
        // if the user hasn't already closed the panel while this was in flight.
        if (panel === openPanel) positionDropdown(panel, toggle);
      })
      .catch(() => {
        panel.innerHTML = '<div class="dropdown__empty">Couldn\'t load past questions.</div>';
      });
  }

  // Picking a past question re-opens it exactly like a fresh answer.
  document.body.addEventListener("click", (ev) => {
    const item = ev.target.closest("[data-open-search-report]");
    if (!item) return;
    closeDropdown();
    openSearchReport(item.dataset.runId, item.dataset.filename);
  });

  // Deleting a past question, from the trash icon in its history row.
  // Two clicks, no popup: the first turns the icon into an inline "confirm
  // delete?" for a few seconds (quiet, matches the rest of this dropdown's
  // tone) rather than a window.confirm() or a toast with a button. A click
  // anywhere else, or letting it time out, cancels back to the plain icon.
  const DELETE_CONFIRM_MS = 3000;
  let pendingDelete = null; // the button currently showing its confirm state

  function resetPendingDelete() {
    if (!pendingDelete) return;
    clearTimeout(pendingDelete.timer);
    pendingDelete.btn.classList.remove("is-confirming");
    pendingDelete.btn.innerHTML = pendingDelete.btn.dataset.originalIcon;
    pendingDelete.btn.title = "Delete this question";
    pendingDelete.btn.setAttribute("aria-label", "Delete this question");
    // Clicking the button leaves it focused, and a focused button keeps
    // the row's :focus-within background (and the button's own
    // :focus-visible opacity) engaged even once the pointer has moved
    // away. Blur it so cancelling — timeout, click elsewhere, or the
    // mouseleave below — actually returns the row to its plain,
    // unhovered look instead of staying stuck looking "armed".
    pendingDelete.btn.blur();
    pendingDelete = null;
  }

  document.body.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-delete-search]");
    if (!btn) {
      // Any other click in the document cancels an in-progress confirm.
      resetPendingDelete();
      return;
    }
    ev.preventDefault();
    ev.stopPropagation(); // never let this reach the question button underneath

    if (pendingDelete && pendingDelete.btn === btn) {
      // Second click within the window: actually delete.
      clearTimeout(pendingDelete.timer);
      pendingDelete = null;
      const { runId, filename } = btn.dataset;
      const row = btn.closest(".dropdown__item--search");
      const panel = btn.closest("[data-search-history]");
      fetch(`/api/runs/${encodeURIComponent(runId)}/searches/${encodeURIComponent(filename)}`,
            { method: "DELETE" })
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          if (row) row.remove();
          if (panel && !panel.querySelector(".dropdown__item--search")) {
            panel.innerHTML = '<div class="dropdown__empty">No questions asked yet for this run.</div>';
          }
        })
        .catch(() => showToast("Couldn't delete that question."));
      return;
    }

    // First click: arm it, and reset any other row's pending confirm.
    resetPendingDelete();
    btn.dataset.originalIcon = btn.innerHTML;
    btn.innerHTML =
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
      'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<path d="M5.2 12.4 9.9 17.1 18.8 6.9"/></svg>';
    btn.classList.add("is-confirming");
    btn.title = "Click again to permanently delete";
    btn.setAttribute("aria-label", "Click again to permanently delete");
    pendingDelete = { btn, timer: setTimeout(resetPendingDelete, DELETE_CONFIRM_MS) };

    // Leaving the row — moving to another title, or off the dropdown
    // entirely — cancels right away instead of waiting for the timeout,
    // same as how every other row's hover state behaves. Click-elsewhere
    // and the timeout above stay as fallbacks for touch, where there's
    // no hover to leave.
    const row = btn.closest(".dropdown__item--search");
    if (row) row.addEventListener("mouseleave", resetPendingDelete, { once: true });
  });

  // "[\[E12\]](#cite-E12)" links land here; the matching "## Sources" entry
  // carries the "cite-E12" id inside its own heading (see renderMarkdown's
  // anchor splice above). A plain in-page "#anchor" link from the model's
  // own Markdown falls through the same handler.
  searchReportBody.addEventListener("click", (ev) => {
    const link = ev.target.closest("[data-panel-jump]");
    if (!link) return;
    ev.preventDefault();
    const anchor = document.getElementById(link.getAttribute("href").slice(1));
    if (!anchor) return;
    // render_citation_appendix() (search.py) writes the anchor *inside* the
    // entity's own heading — "### <a id=\"cite-E12\"></a>[E12] Title" — so
    // the heading itself is the thing to scroll to and flash.
    const flashTarget = anchor.closest("h2, h3, h4, h5") || anchor;
    // "start", not "center" — a plain anchor jump in any .md viewer lands
    // the target near the top, not mid-screen; scroll-margin-top on the
    // heading (dashboard.css) keeps it from sitting flush against the edge.
    flashTarget.scrollIntoView({ behavior: "smooth", block: "start" });
    flashTarget.classList.remove("report-cite-target--flash");
    void flashTarget.offsetWidth; // restart the animation on a repeat click
    flashTarget.classList.add("report-cite-target--flash");
    flashTarget.addEventListener(
      "animationend",
      () => flashTarget.classList.remove("report-cite-target--flash"),
      { once: true }
    );
  });

  function openDrawer(launchMsg) {
    if (!cfg.ptySupported) {
      showToast("Interactive sessions need a POSIX pty and aren't available on this platform.");
      return false;
    }
    if (sessionLive) {
      showToast("A session is already running in the terminal panel below — close it first.");
      return false;
    }
    hasLaunched = true;
    currentLaunch = launchMsg;
    if (launchMsg.action === "search" && launchMsg.run_id) {
      currentLaunch.knownSearchFiles = listSearchFilenames(launchMsg.run_id);
    }
    ensureTerm();
    term.reset();
    logChunks = [];
    applyStoredDrawerHeight();
    // A search shows its own progress in the report panel (see
    // showSearchProgress) instead of the raw terminal — the session still
    // runs underneath exactly the same way, and the logs still stream into
    // the terminal buffer regardless. The drawer itself must stay closed
    // for a search though (even if it was left open from an earlier
    // resume/relabel/etc.), so force it shut here instead of just skipping
    // the "open" call.
    if (launchMsg.action === "search") {
      drawer.classList.remove("is-open");
      drawer.style.height = "";
      setConsoleToggleUI(false);
      safeFit();
    } else {
      drawer.classList.add("is-open");
      setConsoleToggleUI(true);
      fitWhenSettled();
    }

    argvLabel.textContent = "";
    setStatus("running", "connecting…");

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${proto}//${location.host}/ws/terminal`);
    sessionLive = true;

    socket.addEventListener("open", () => {
      socket.send(JSON.stringify(launchMsg));
    });

    socket.addEventListener("message", (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }

      if (msg.type === "started") {
        setStatus("running", "running");
        argvLabel.textContent = (msg.argv || []).join(" ");
        safeFit();
      } else if (msg.type === "stdout") {
        logChunks.push(msg.data);
        term.write(msg.data);
        if (currentLaunch && currentLaunch.action === "search") updateSearchProgressFromLog();
      } else if (msg.type === "error") {
        setStatus("error", "error");
        term.write(`\r\n\x1b[31m[dashboard] ${msg.message}\x1b[0m\r\n`);
        showToast(msg.message);
        if (currentLaunch && currentLaunch.action === "search") failSearchAndShowLogs();
      } else if (msg.type === "exit") {
        sessionLive = false;
        const wasSearch = currentLaunch && currentLaunch.action === "search" && currentLaunch.run_id;
        if (msg.code === 0) {
          setStatus("done", "finished");
          if (wasSearch) resolveSearchOutcome(currentLaunch);
        } else {
          setStatus("error", `exited (${msg.code})`);
          if (wasSearch) failSearchAndShowLogs(`Search failed (exit ${msg.code}) — see logs.`);
        }
        refreshTables();
      }
    });

    socket.addEventListener("close", () => {
      const wasLive = sessionLive;
      sessionLive = false;
      if (statusText.textContent === "running" || statusText.textContent === "connecting…") {
        setStatus("error", "disconnected");
        if (wasLive && currentLaunch && currentLaunch.action === "search") {
          failSearchAndShowLogs("Connection to the search process was lost — see logs.");
        }
      }
    });

    socket.addEventListener("error", () => {
      setStatus("error", "connection error");
      if (currentLaunch && currentLaunch.action === "search") failSearchAndShowLogs();
    });
    return true;
  }

  document.getElementById("term-close").addEventListener("click", () => {
    if (sessionLive && !window.confirm("Stop the running process? This sends it a termination signal.")) return;
    if (socket) socket.close();
    drawer.classList.remove("is-open");
    drawer.style.height = "";
    sessionLive = false;
    setConsoleToggleUI(false);
  });

  // Only shows/hides the panel — never touches the process. "Stop" inside
  // the panel is the only control that does that.
  function toggleDrawer() {
    if (drawer.classList.contains("is-open")) {
      drawer.classList.remove("is-open");
      drawer.style.height = "";
      setConsoleToggleUI(false);
      return;
    }
    ensureTerm();
    if (!hasLaunched) setStatus("idle", "idle");
    applyStoredDrawerHeight();
    drawer.classList.add("is-open");
    setConsoleToggleUI(true);
    fitWhenSettled();
  }
  consoleToggle.addEventListener("click", toggleDrawer);

  function rowOverrides(runId) {
    const runModeEl = document.getElementById(`run-mode-${runId}`);
    const resumeFromEl = document.getElementById(`resume-from-${runId}`);
    return {
      run_mode: runModeEl && runModeEl.value ? runModeEl.value : null,
      resume_from: resumeFromEl && resumeFromEl.value ? resumeFromEl.value : null,
    };
  }

  document.body.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-launch]");
    if (!btn) return;
    if (btn.dataset.confirm && !window.confirm(btn.dataset.confirm)) return;

    const action = btn.dataset.launch;
    const runId = btn.dataset.runId;
    const ov = rowOverrides(runId);
    closeDropdown();

    if (action === "resume") {
      openDrawer({ action, run_id: runId, run_mode: ov.run_mode, resume_from: ov.resume_from });
    } else if (action === "relabel") {
      openDrawer({ action, run_id: runId, force_skip_llm: btn.dataset.forceSkipLlm === "true" });
    } else if (action === "show_render") {
      openDrawer({ action, run_id: runId });
    } else if (action === "sync_models") {
      openDrawer({ action, run_id: runId });
    } else if (action === "search") {
      const input = document.getElementById(`search-query-${runId}`);
      const query = input ? input.value.trim() : "";
      if (!query) { showToast("Type a question first."); return; }
      openDrawer({ action, run_id: runId, query });
    }
  });

  // Position: fixed with computed coordinates, not CSS anchoring — a row
  // can sit anywhere in the scrolling table. Doesn't close on a background
  // refresh; see the Idiomorph callbacks further down for why.
  let openPanel = null;
  let openToggle = null;

  // pointerdown, not click: a native <select>'s option list doesn't
  // consistently report the panel as the click target across browsers.
  let pointerDownInsidePanel = false;

  document.addEventListener("pointerdown", (ev) => {
    if (!openPanel) return;
    pointerDownInsidePanel = !!(
      ev.target.closest(".dropdown__panel") || ev.target.closest("[data-dropdown-toggle]")
    );
  }, true);

  function positionDropdown(panel, toggle) {
    const margin = 8;
    const gap = 4;
    const toggleRect = toggle.getBoundingClientRect();
    const panelRect = panel.getBoundingClientRect();
    const viewportW = document.documentElement.clientWidth;
    const viewportH = document.documentElement.clientHeight;

    let left = toggleRect.right - panelRect.width;
    left = Math.max(margin, Math.min(left, viewportW - panelRect.width - margin));

    const spaceBelow = viewportH - toggleRect.bottom;
    const spaceAbove = toggleRect.top;
    const opensUp = spaceBelow < panelRect.height + gap + margin && spaceAbove > spaceBelow;

    let top = opensUp
      ? toggleRect.top - panelRect.height - gap
      : toggleRect.bottom + gap;
    top = Math.max(margin, Math.min(top, viewportH - panelRect.height - margin));

    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
  }

  // So reopening a panel starts fresh, not from a previous choice.
  function resetPanelFields(panel) {
    panel.querySelectorAll("select").forEach((el) => { el.selectedIndex = 0; });
    panel.querySelectorAll("input").forEach((el) => {
      el.value = el.defaultValue;
      el.checked = el.defaultChecked;
    });
  }

  function closeDropdown() {
    if (!openPanel) return;
    resetPanelFields(openPanel);
    openPanel.hidden = true;
    if (openToggle) openToggle.setAttribute("aria-expanded", "false");
    openPanel = null;
    openToggle = null;
  }

  function openDropdown(panel, toggle) {
    if (openPanel && openPanel !== panel) closeDropdown();
    panel.hidden = false;
    toggle.setAttribute("aria-expanded", "true");
    openPanel = panel;
    openToggle = toggle;
    positionDropdown(panel, toggle);
  }

  document.body.addEventListener("click", (ev) => {
    const toggle = ev.target.closest("[data-dropdown-toggle]");
    if (toggle) {
      const panel = document.getElementById(toggle.getAttribute("aria-controls"));
      if (!panel) return;
      if (panel === openPanel) {
        closeDropdown();
      } else {
        openDropdown(panel, toggle);
        if (panel.hasAttribute("data-search-history")) loadSearchHistory(panel, toggle);
      }
      return;
    }
    if (pointerDownInsidePanel) return;
    if (!ev.target.closest(".dropdown__panel")) closeDropdown();
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape" || !openPanel) return;
    const toggle = openToggle;
    closeDropdown();
    if (toggle) toggle.focus();
  });

  // Capture phase: catches scroll on any scrollable ancestor
  // (.runs-table-scroll included), which doesn't otherwise bubble.
  document.addEventListener("scroll", () => {
    if (openPanel && openToggle) positionDropdown(openPanel, openToggle);
  }, true);
  window.addEventListener("resize", () => {
    if (openPanel && openToggle) positionDropdown(openPanel, openToggle);
  });

  // hx-preserve doesn't apply to morph:innerHTML swaps (Idiomorph), so
  // without these hooks any background refresh would morph an open panel
  // straight back to _runs_table.html's `hidden` markup.
  const activePanel = document.getElementById("panel-active");

  Idiomorph.defaults.callbacks.beforeNodeMorphed = function (oldNode) {
    // Skipping the panel's own node also skips its <select>s underneath —
    // Idiomorph never recurses into a node it's told not to morph.
    return !(openPanel && oldNode === openPanel);
  };

  Idiomorph.defaults.callbacks.beforeAttributeUpdated = function (attrName, node) {
    // Matched by aria-controls, not by reference — the toggle carries no
    // id of its own for Idiomorph to key on.
    if (attrName !== "aria-expanded" || !openPanel) return true;
    return node.getAttribute("aria-controls") !== openPanel.id;
  };

  // Neither hook fires for a node removed outright (run trashed/purged
  // elsewhere) — treat a disconnected openPanel as closed after settle.
  activePanel.addEventListener("htmx:afterSettle", () => {
    if (openPanel && !openPanel.isConnected) {
      openPanel = null;
      openToggle = null;
    }
  });

  const newRunModal = document.getElementById("modal-new-run");
  const newRunPreview = document.getElementById("new-run-preview");
  const fRunMode = document.getElementById("f-run-mode");
  const fContentType = document.getElementById("f-content-type");
  const fInput = document.getElementById("f-input");
  const fSetK = document.getElementById("f-set-k");

  // Display only — must stay in lockstep with runs_api.build_new_run_argv()
  // or the preview lies. --fresh isn't a field: this button never resumes.
  function updateNewRunPreview() {
    const parts = ["python", "pipeline.py"];
    if (fRunMode.value) parts.push("--run-mode", fRunMode.value);
    if (fContentType.value) parts.push("--content-type", fContentType.value);
    const inputDir = fInput.value.trim();
    if (inputDir) parts.push("--input", inputDir);
    if (fSetK.value) parts.push("--set-k", fSetK.value);
    parts.push("--fresh");
    newRunPreview.textContent = parts.join(" ");
  }
  [fRunMode, fContentType, fInput, fSetK].forEach((el) => {
    el.addEventListener("input", updateNewRunPreview);
    el.addEventListener("change", updateNewRunPreview);
  });

  function resetNewRunModal() {
    newRunModal.querySelectorAll("select").forEach((el) => { el.selectedIndex = 0; });
    newRunModal.querySelectorAll("input").forEach((el) => {
      el.value = el.defaultValue;
      el.checked = el.defaultChecked;
    });
    updateNewRunPreview();
  }
  newRunModal._resetFields = resetNewRunModal;

  document.getElementById("btn-new-run").addEventListener("click", () => {
    updateNewRunPreview();
    openBackdropModal(newRunModal);
  });
  document.getElementById("modal-new-run-cancel").addEventListener("click", () => {
    closeBackdropModal(newRunModal);
  });
  newRunModal.addEventListener("click", (ev) => {
    if (ev.target === newRunModal) closeBackdropModal(newRunModal);
  });
  document.getElementById("modal-new-run-launch").addEventListener("click", () => {
    const runMode = fRunMode.value || null;
    const contentType = fContentType.value || null;
    const inputDir = fInput.value.trim() || null;
    const setKRaw = fSetK.value;
    const setK = setKRaw ? parseInt(setKRaw, 10) : null;
    closeBackdropModal(newRunModal);
    openDrawer({ action: "new_run", run_mode: runMode, content_type: contentType, input_dir: inputDir, set_k: setK });
  });

  const searchModal = document.getElementById("modal-search");
  const searchModalInput = document.getElementById("search-modal-input");
  const searchModalBudget = document.getElementById("search-modal-budget");
  const searchModalBudgetValue = document.getElementById("search-modal-budget-value");
  const SEARCH_BUDGET_DEFAULT = 3000;

  function closeSearchModal() {
    closeBackdropModal(searchModal);
  }

  searchModalBudget.addEventListener("input", () => {
    searchModalBudgetValue.textContent = searchModalBudget.value;
  });

  document.body.addEventListener("click", (ev) => {
    const trigger = ev.target.closest("[data-open-search-modal]");
    if (!trigger) return;
    searchModal.dataset.runId = trigger.dataset.runId;
    searchModalInput.value = "";
    searchModalBudget.value = SEARCH_BUDGET_DEFAULT;
    searchModalBudgetValue.textContent = String(SEARCH_BUDGET_DEFAULT);
    openBackdropModal(searchModal);
    setTimeout(() => searchModalInput.focus(), 0);
  });
  document.getElementById("modal-search-cancel").addEventListener("click", closeSearchModal);
  searchModal.addEventListener("click", (ev) => {
    if (ev.target === searchModal) closeSearchModal();
  });

  function submitSearchModal() {
    const query = searchModalInput.value.trim();
    if (!query) { showToast("Type a question first."); return; }
    const runId = searchModal.dataset.runId;
    const budget = parseInt(searchModalBudget.value, 10) || SEARCH_BUDGET_DEFAULT;
    closeSearchModal();
    if (openDrawer({ action: "search", run_id: runId, query, budget })) {
      showSearchProgress(query);
    }
  }
  document.getElementById("modal-search-submit").addEventListener("click", submitSearchModal);
  searchModalInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") submitSearchModal();
  });

  // <input type=file webkitdirectory> never exposes an absolute path to
  // JS, and pipeline.py needs the real path server-side — so this walks
  // GET /api/browse instead (runs_api.py's browse_directory()).
  const browseModal = document.getElementById("modal-browse");
  const browsePathInput = document.getElementById("browse-path-input");
  const browseUpBtn = document.getElementById("browse-up");
  const browseList = document.getElementById("browse-list");
  const browseMeta = document.getElementById("browse-meta");
  const browseSelectBtn = document.getElementById("modal-browse-select");

  let browseCurrentPath = null;
  let browseParentPath = null;
  // Guards against a slow response landing after a faster, later one and
  // reverting the picker to an earlier folder.
  let browseRequestToken = 0;

  // textContent/createElement, not innerHTML — folder names come from the
  // filesystem and could contain markup-like characters.
  function browseErrorNode(message) {
    const el = document.createElement("div");
    el.className = "browse__error";
    el.textContent = message;
    return el;
  }

  async function loadBrowseDir(path) {
    const token = ++browseRequestToken;
    browseList.innerHTML = '<div class="empty-state"><strong>Loading…</strong></div>';
    browseMeta.textContent = "";
    browseMeta.className = "browse__meta";

    let res, data;
    try {
      const qs = path ? `?path=${encodeURIComponent(path)}` : "";
      res = await fetch(`/api/browse${qs}`);
      data = await res.json();
    } catch (err) {
      if (token !== browseRequestToken) return;
      browseList.innerHTML = "";
      browseList.appendChild(browseErrorNode(`Network error: ${err.message}`));
      return;
    }
    if (token !== browseRequestToken) return;

    if (!res.ok) {
      browseList.innerHTML = "";
      browseList.appendChild(browseErrorNode(data.detail || "Couldn't open that folder."));
      return;
    }

    browseCurrentPath = data.path;
    browseParentPath = data.parent;
    browsePathInput.value = data.path;
    browseUpBtn.disabled = !data.parent;

    browseList.innerHTML = "";
    if (data.dirs.length === 0 && data.files.length === 0) {
      browseList.innerHTML = '<div class="empty-state"><strong>Empty folder</strong>Select this folder itself below, or go up.</div>';
    } else {
      for (const d of data.dirs) {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "browse__row";

        const icon = document.createElement("span");
        icon.className = "browse__row-icon";
        icon.setAttribute("aria-hidden", "true");
        icon.textContent = "\u25B8";
        row.appendChild(icon);

        const label = document.createElement("span");
        label.textContent = d.name;
        row.appendChild(label);

        row.addEventListener("click", () => loadBrowseDir(d.path));
        browseList.appendChild(row);
      }

      // Plain rows, not buttons — --input always takes a folder, never a
      // file (see browse_directory()).
      for (const f of data.files) {
        const row = document.createElement("div");
        row.className = "browse__row browse__row--file";

        const icon = document.createElement("span");
        icon.className = "browse__row-icon";
        icon.setAttribute("aria-hidden", "true");
        icon.textContent = "\u2013";
        row.appendChild(icon);

        const label = document.createElement("span");
        label.className = "browse__row-label";
        label.textContent = f.name;
        row.appendChild(label);

        const size = document.createElement("span");
        size.className = "browse__row-size";
        size.textContent = formatFileSize(f.size);
        row.appendChild(size);

        browseList.appendChild(row);
      }
    }

    if (data.file_count > 0) {
      browseMeta.textContent =
        `${data.file_count} supported file${data.file_count === 1 ? "" : "s"} directly in this folder — a valid input folder.`;
      browseMeta.classList.add("browse__meta--has-files");
    } else {
      browseMeta.textContent =
        "No supported files directly in this folder (.json/.csv/.tsv/.txt/.md/.pdf) — check a subfolder, or the right one if files are nested deeper.";
      browseMeta.classList.add("browse__meta--empty");
    }
  }

  function formatFileSize(bytes) {
    if (bytes === null || bytes === undefined) return "";
    if (bytes < 1024) return `${bytes} B`;
    const units = ["KB", "MB", "GB", "TB"];
    let value = bytes / 1024;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }
    return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unitIndex]}`;
  }

  document.getElementById("f-input-browse").addEventListener("click", () => {
    openBackdropModal(browseModal);
    loadBrowseDir(fInput.value.trim() || null);
  });
  document.getElementById("modal-browse-cancel").addEventListener("click", () => {
    closeBackdropModal(browseModal);
  });
  browseModal.addEventListener("click", (ev) => {
    if (ev.target === browseModal) closeBackdropModal(browseModal);
  });
  browseUpBtn.addEventListener("click", () => {
    if (browseParentPath) loadBrowseDir(browseParentPath);
  });
  browsePathInput.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter") return;
    loadBrowseDir(browsePathInput.value.trim());
  });
  browseSelectBtn.addEventListener("click", () => {
    if (!browseCurrentPath) return;
    fInput.value = browseCurrentPath;
    updateNewRunPreview();
    closeBackdropModal(browseModal);
  });

  // htmx (_runs_table.html's hx-get/hx-target) owns fetching and swapping
  // into #detail-panel-body; this only owns the panel's chrome.
  const detailBackdrop = document.getElementById("detail-panel-backdrop");
  const detailBody = document.getElementById("detail-panel-body");

  function openDetailPanel() {
    closeDropdown();
    detailBackdrop.classList.add("is-open");
  }
  function closeDetailPanel() {
    detailBackdrop.classList.remove("is-open");
  }

  document.body.addEventListener("click", (ev) => {
    if (!ev.target.closest("[data-open-detail]")) return;
    // Set synchronously on click, not left to wait for htmx's response:
    // without this, opening a second run_id while the panel still shows
    // the previous one's content would flash stale data for a moment.
    detailBody.innerHTML = '<div class="empty-state"><strong>Loading…</strong></div>';
    openDetailPanel();
  });

  document.getElementById("detail-panel-close").addEventListener("click", closeDetailPanel);
  detailBackdrop.addEventListener("click", (ev) => {
    if (ev.target === detailBackdrop) closeDetailPanel();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && detailBackdrop.classList.contains("is-open")) closeDetailPanel();
  });

  // A fresh swap always starts scrolled to the top — otherwise opening a
  // second run_id while scrolled down into the first one's config block
  // would land already scrolled past the new run's own header.
  detailBody.addEventListener("htmx:afterSwap", (ev) => {
    if (ev.target === detailBody) detailBody.scrollTop = 0;
  });

  // Copies whatever data-copy-target names, not the button's own text.
  document.body.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-copy-target]");
    if (!btn) return;
    const target = document.getElementById(btn.dataset.copyTarget);
    if (!target || !navigator.clipboard) return;
    navigator.clipboard.writeText(target.textContent).then(() => {
      const original = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = original; }, 1500);
    }).catch(() => showToast("Couldn't copy to clipboard."));
  });

})();