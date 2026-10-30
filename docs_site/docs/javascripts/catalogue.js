/* Progressive enhancement: the static 24-row catalogue stays readable without JS. */
(function () {
  function init() {
    const host = document.getElementById("gragra-catalogue-controls");
    if (!host || host.dataset.initialized) return;
    host.dataset.initialized = "1";
    const tables = Array.from(document.querySelectorAll(".gragra-catalogue-table"));
    const rows = tables.flatMap(t => Array.from(t.querySelectorAll("tbody tr")));
    const paths = new Map(rows.map(r => [r, JSON.parse(r.dataset.gragraPaths)]));
    const cats = [...new Set(rows.map(r => r.dataset.gragraCategory))];
    const labels = {static:"Static", phasor:"Phasor", time_series:"Time series",
      fourier_series:"Fourier series", fourier_transform:"Fourier transform",
      spectral_density:"Spectral density", other:"Fourier amplitude (see API)"};
    function formatKey(o) { return o.representation + (o.statistic ? ":" + o.statistic : ""); }
    function formatLabel(o) { return labels[o.representation] + (o.statistic ? " (" + o.statistic.toUpperCase() + ")" : ""); }
    const formats = new Map(rows.flatMap(r => paths.get(r).flatMap(p =>
      p.outputs.map(o => [formatKey(o), formatLabel(o)]))));
    host.innerHTML = '<div class="gragra-filters" role="search" aria-label="Filter models">' +
      '<label>Search models<input type="search" data-role="search" placeholder="Model or API" aria-label="Search models"></label>' +
      '<label>Category<select data-role="category"><option value="">All categories</option></select></label>' +
      '<label>Representation<select data-role="format"><option value="">All representations</option></select></label>' +
      '<button type="button" data-role="clear">Clear filters</button>' +
      '<span class="gragra-count" role="status" aria-live="polite"></span></div>' +
      '<p class="gragra-empty" role="status">No models match these filters.</p>';
    const q = host.querySelector('[data-role="search"]');
    const category = host.querySelector('[data-role="category"]');
    const format = host.querySelector('[data-role="format"]');
    const status = host.querySelector(".gragra-count");
    const empty = host.querySelector(".gragra-empty");
    function add(select, value, label) { const o = document.createElement("option"); o.value=value; o.textContent=label; select.appendChild(o); }
    cats.forEach(v => add(category, v, v.replace(/_/g, " ")));
    formats.forEach((label, value) => add(format, value, label));
    function filter() {
      const term=q.value.trim().toLowerCase();
      let visible=0;
      rows.forEach(r => {
        const common=(r.dataset.gragraSearch||"");
        const matchesPath=paths.get(r).some(p => {
          const words=[common,p.label,...p.api_symbols,...p.outputs.map(formatLabel)].join(" ").toLowerCase();
          return (!term||words.includes(term))&&(!format.value||p.outputs.some(o=>formatKey(o)===format.value));
        });
        const match=matchesPath&&(!category.value||r.dataset.gragraCategory===category.value);
        r.hidden=!match; if(match)visible++;
      });
      tables.forEach(t => {const section=t.previousElementSibling;const any=Array.from(t.querySelectorAll("tbody tr")).some(r=>!r.hidden);t.hidden=!any;if(section&&/^H[2-6]$/.test(section.tagName))section.hidden=!any;});
      status.textContent=visible+" of "+rows.length+" model groups";
      empty.style.display=visible?"none":"block";
    }
    q.addEventListener("input",filter); category.addEventListener("change",filter); format.addEventListener("change",filter);
    host.querySelector('[data-role="clear"]').addEventListener("click",() => {q.value="";category.value="";format.value="";filter();q.focus();});
    filter();
  }
  if(document.readyState==="loading") document.addEventListener("DOMContentLoaded",init);
  else init();
  if(typeof document$!=="undefined") document$.subscribe(init);
})();
