/* Progressive enhancement: the static 24-row catalogue stays readable without JS. */
(function () {
  function init() {
    const host = document.getElementById("gragra-catalogue-controls");
    if (!host || host.dataset.initialized) return;
    host.dataset.initialized = "1";
    const tables = Array.from(document.querySelectorAll(".gragra-catalogue-table"));
    const rows = tables.flatMap(t => Array.from(t.querySelectorAll("tbody tr")));
    const cats = [...new Set(rows.map(r => r.dataset.gragraCategory))];
    const formats = [...new Set(rows.flatMap(r => (r.dataset.gragraFormats || "").split(" ").filter(Boolean)))];
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
    formats.forEach(v => add(format, v, v.replace(/_/g, " ")));
    function filter() {
      const term=q.value.trim().toLowerCase();
      let visible=0;
      rows.forEach(r => {
        const words=(r.dataset.gragraSearch||"");
        const fmts=(r.dataset.gragraFormats||"").split(" ");
        const match=(!term||words.includes(term))&&(!category.value||r.dataset.gragraCategory===category.value)&&(!format.value||fmts.includes(format.value));
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
