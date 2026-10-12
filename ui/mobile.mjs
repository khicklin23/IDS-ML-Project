// Shared touch and keyboard support, including dynamically rendered metric cards.
export function installMobileSupport(doc = globalThis.document, win = globalThis.window) {
  if (!doc?.addEventListener || !win) return;
  let active = null;
  const close = () => {
    if (!active) return;
    active.classList.remove('help-open');
    active.setAttribute('aria-expanded', 'false');
    active = null;
  };
  const position = button => {
    const tip = button.querySelector('[role="tooltip"]');
    if (!tip) return;
    const viewport = win.visualViewport;
    const width = viewport?.width ?? win.innerWidth;
    const height = viewport?.height ?? win.innerHeight;
    const left = viewport?.offsetLeft ?? 0;
    const top = viewport?.offsetTop ?? 0;
    const box = button.getBoundingClientRect();
    const tipWidth = Math.min(280, width - 32);
    tip.style.width = `${tipWidth}px`;
    tip.style.left = `${Math.max(left + 16, Math.min(box.left, left + width - tipWidth - 16))}px`;
    tip.style.top = `${Math.max(top + 16, Math.min(box.bottom + 8, top + height - tip.offsetHeight - 16))}px`;
  };
  const open = button => {
    close();
    active = button;
    button.classList.remove('help-dismissed');
    button.classList.add('help-open');
    button.setAttribute('aria-expanded', 'true');
    position(button);
  };
  doc.addEventListener('click', event => {
    const button = event.target.closest?.('.metric-info');
    if (button) { if (button === active) close(); else open(button); }
    else close();
  });
  doc.addEventListener('pointerover', event => {
    if (event.pointerType === 'touch') return;
    const button = event.target.closest?.('.metric-info');
    if (button && button !== active) open(button);
  });
  doc.addEventListener('pointerout', event => {
    if (event.pointerType === 'touch') return;
    if (active && event.target.closest?.('.metric-info') === active && !active.contains(event.relatedTarget)) close();
  });
  doc.addEventListener('focusin', event => {
    // Touch clicks control their own toggle; keyboard focus is visible immediately.
    if (event.target.matches?.('.metric-info:focus-visible')) open(event.target);
    else if (!event.target.closest?.('.metric-info')) close();
  });
  doc.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });
  doc.addEventListener('scroll', close, true);
  win.addEventListener?.('resize', close);
  win.visualViewport?.addEventListener('resize', close);
  doc.querySelectorAll('.table-wrap, .version-table-wrap, .matrix-wrap, .network-scroll').forEach(region => {
    region.setAttribute('tabindex', '0');
    region.setAttribute('role', 'region');
    region.setAttribute('aria-label', region.classList.contains('network-scroll') ? 'Neural network diagram. Scroll horizontally to explore all four stages.' : 'Data table. Scroll horizontally to see more columns.');
  });
}
