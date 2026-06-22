var rows = Array.isArray(DATA) ? DATA : (DATA && DATA.results ? DATA.results : []);
var byFile = {};
for (var i = 0; i < rows.length; i++) {
  var r = rows[i];
  var f = r.file || "?";
  if (!byFile[f]) byFile[f] = [];
  if (r.name) byFile[f].push(r.name);
}
return Object.keys(byFile).map(function(f) {
  var names = byFile[f];
  return { file: f, count: names.length, names: names.slice(0, 12) };
});
