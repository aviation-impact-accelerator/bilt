(() => {
  "use strict";

  console.log("Mermaid zoom loaded");

  let activeOverlay = null;

  function openDiagram(mermaid) {
    if (activeOverlay) {
      return;
    }

    // Remember exactly where the diagram came from.
    const placeholder = document.createComment("mermaid-zoom-placeholder");
    mermaid.before(placeholder);

    const overlay = document.createElement("div");
    overlay.className = "mermaid-zoom-overlay";

    const toolbar = document.createElement("div");
    toolbar.className = "mermaid-zoom-toolbar";

    const hint = document.createElement("span");
    hint.textContent =
      "Scroll to zoom · drag to pan · double-click to reset · Esc to close";

    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = "Reset";

    const close = document.createElement("button");
    close.type = "button";
    close.className = "mermaid-zoom-close";
    close.textContent = "×";
    close.setAttribute("aria-label", "Close diagram");

    toolbar.appendChild(hint);
    toolbar.appendChild(reset);
    toolbar.appendChild(close);

    const viewport = document.createElement("div");
    viewport.className = "mermaid-zoom-viewport";

    const stage = document.createElement("div");
    stage.className = "mermaid-zoom-stage";

    // Move the ACTUAL Mermaid element.
    // Do not clone it: Zensical's rendered content isn't in the normal DOM.
    stage.appendChild(mermaid);
    viewport.appendChild(stage);

    overlay.appendChild(toolbar);
    overlay.appendChild(viewport);
    document.body.appendChild(overlay);

    document.body.classList.add("mermaid-zoom-open");

    activeOverlay = overlay;

    let scale = 1;
    let x = 0;
    let y = 0;

    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function update() {
      stage.style.transform =
        `translate(${x}px, ${y}px) scale(${scale})`;
    }

    function resetView() {
      scale = 1;
      x = 0;
      y = 0;
      update();
    }

    function closeDiagram() {
      if (!activeOverlay) {
        return;
      }

      // Put the actual Mermaid element back where it came from.
      placeholder.replaceWith(mermaid);

      overlay.remove();
      document.body.classList.remove("mermaid-zoom-open");

      document.removeEventListener("keydown", handleKeydown);

      activeOverlay = null;
    }

    function handleKeydown(event) {
      if (event.key === "Escape") {
        closeDiagram();
      }
    }

    viewport.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();

        const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;

        scale = Math.max(
          0.5,
          Math.min(8, scale * factor),
        );

        update();
      },
      { passive: false },
    );

    viewport.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return;
      }

      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;

      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add("is-dragging");
    });

    viewport.addEventListener("pointermove", (event) => {
      if (!dragging) {
        return;
      }

      x += event.clientX - lastX;
      y += event.clientY - lastY;

      lastX = event.clientX;
      lastY = event.clientY;

      update();
    });

    function stopDragging(event) {
      if (!dragging) {
        return;
      }

      dragging = false;

      if (viewport.hasPointerCapture(event.pointerId)) {
        viewport.releasePointerCapture(event.pointerId);
      }

      viewport.classList.remove("is-dragging");
    }

    viewport.addEventListener("pointerup", stopDragging);
    viewport.addEventListener("pointercancel", stopDragging);

    viewport.addEventListener("dblclick", (event) => {
      event.preventDefault();
      resetView();
    });

    reset.addEventListener("click", (event) => {
      event.stopPropagation();
      resetView();
    });

    close.addEventListener("click", (event) => {
      event.stopPropagation();
      closeDiagram();
    });

    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) {
        closeDiagram();
      }
    });

    document.addEventListener("keydown", handleKeydown);

    resetView();
  }

  /*
   * Listen for the Mermaid HOST element itself.
   *
   * This deliberately does not look for ".mermaid svg":
   * Zensical's rendered diagram is not present as normal children of
   * the .mermaid element.
   */
  document.addEventListener("click", (event) => {
    if (activeOverlay) {
      return;
    }

    if (!(event.target instanceof Element)) {
      return;
    }

    const mermaid = event.target.closest(".mermaid");

    if (!mermaid) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    openDiagram(mermaid);
  });
})();