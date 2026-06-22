var rows = Array.isArray(DATA) ? DATA : (DATA && DATA.results ? DATA.results : []);
var byKind = {};
for (var i = 0; i < rows.length; i++) {
  var r = rows[i];
  var k = r.kind || "unknown";
  if (!byKind[k]) byKind[k] = [];
  byKind[k].push({ name: r.name, file: r.file });
}
return Object.keys(byKind).map(function(k) {
  return { kind: k, count: byKind[k].length, symbols: byKind[k].slice(0, 15) };
});
