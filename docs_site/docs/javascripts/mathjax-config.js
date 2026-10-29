/* MathJax v3 configuration for the gragra documentation site.
 *
 * Loaded *before* javascripts/mathjax/tex-mml-chtml.js (see `extra_javascript`
 * in mkdocs.yml -- plain, non-async script tags execute in document order, so
 * `window.MathJax` is already populated when the component boots).
 *
 * The delimiters below match what `pymdownx.arithmatex` emits in `generic`
 * mode. The `document$` subscription re-typesets after Material's instant
 * navigation swaps page content in without a full reload.
 *
 * No third-party host is contacted: the component and its CHTML web fonts are
 * vendored under javascripts/mathjax/ (MathJax 3.2.2, Apache-2.0). MathJax
 * resolves the font directory relative to the directory of the script that
 * loaded it, so the vendored layout javascripts/mathjax/output/chtml/fonts/
 * woff-v2/ is found without any extra configuration.
 */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  });
}
