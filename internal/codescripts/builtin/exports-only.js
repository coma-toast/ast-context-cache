var rows = Array.isArray(DATA) ? DATA : (DATA && DATA.results ? DATA.results : []);
return rows.filter(function(r) {
  var n = r.name || "";
  if (!n) return false;
  var c = n.charAt(0);
  return c === c.toUpperCase() && c !== c.toLowerCase();
}).map(function(r) {
  return { name: r.name, kind: r.kind, file: r.file };
});
