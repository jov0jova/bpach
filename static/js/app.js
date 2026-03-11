/**
 * CryptoAlgoFinder – App JS helpers
 */

// ── HTMX event handling ──────────────────────────────────────────
document.addEventListener('htmx:afterSwap', function(evt) {
  // Re-render any Plotly charts in swapped content
  const charts = evt.detail.target.querySelectorAll('[data-plotly]');
  charts.forEach(el => {
    try {
      const data = JSON.parse(el.dataset.plotly);
      Plotly.newPlot(el, data.data, data.layout, {responsive: true, displayModeBar: false});
    } catch(e) { /* ignore */ }
  });
});

// ── Auto-dismiss progress pollers when task is done ─────────────
document.addEventListener('htmx:afterSwap', function(evt) {
  const target = evt.detail.target;
  // If the task is done or errored, stop polling
  if (target.querySelector('.text-success, .text-danger') &&
      !target.querySelector('.spinner-border')) {
    // Remove hx-trigger to stop polling
    target.removeAttribute('hx-trigger');
    target.removeAttribute('hx-get');
  }
});

// ── Confirm dialogs ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  // Auto-focus first input in forms
  const firstInput = document.querySelector('.caf-form input:not([type=hidden]), .caf-form select');
  if (firstInput) firstInput.focus();

  // Tooltips
  const tooltipEls = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltipEls.forEach(el => new bootstrap.Tooltip(el));
});

// ── Number formatting helper ─────────────────────────────────────
function fmt(n, decimals = 2) {
  if (n === null || n === undefined || isNaN(n)) return '—';
  return parseFloat(n).toFixed(decimals);
}

function fmtPct(n, decimals = 1) {
  if (n === null || n === undefined || isNaN(n)) return '—';
  const v = parseFloat(n);
  return (v >= 0 ? '+' : '') + v.toFixed(decimals) + '%';
}

// ── Plotly render helper ─────────────────────────────────────────
function renderPlotly(divId, jsonStr) {
  if (!divId || !jsonStr) return;
  try {
    const data = typeof jsonStr === 'string' ? JSON.parse(jsonStr) : jsonStr;
    Plotly.newPlot(divId, data.data, data.layout, {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
    });
  } catch(e) {
    console.warn('renderPlotly error:', e);
  }
}

// ── Copy to clipboard ────────────────────────────────────────────
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('Copied!');
  });
}

function showToast(msg) {
  const toast = document.createElement('div');
  toast.className = 'position-fixed bottom-0 end-0 m-3 p-2 px-3 rounded bg-success text-white';
  toast.style.zIndex = '9999';
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2000);
}

// ── Keyboard shortcuts ───────────────────────────────────────────
document.addEventListener('keydown', function(e) {
  // Ctrl+S to submit strategy form
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    const submitBtn = document.querySelector('#strategy-form button[type=submit]');
    if (submitBtn) {
      e.preventDefault();
      submitBtn.click();
    }
  }
});
