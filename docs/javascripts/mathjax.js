// Arithmatex prepares the markup; MathJax renders it in the browser.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Initial rendering is handled by MathJax startup. Re-render subsequent
// Zensical page changes, including instant navigation, in sequence.
let mathRendering = Promise.resolve();
document$.subscribe(() => {
  if (!window.MathJax.typesetPromise) return;
  mathRendering = mathRendering.then(() => MathJax.startup.promise).then(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});
