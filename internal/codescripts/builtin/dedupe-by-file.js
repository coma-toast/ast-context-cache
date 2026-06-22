var rows = Array.isArray(DATA) ? DATA : (DATA && DATA.results ? DATA.results : []);
var seen = {};
var out = [];
for (var i = 0; i < rows.length; i++) {
  var r = rows[i];
  var f = r.file || "?";
  if (seen[f]) continue;
  seen[f] = true;
  out.push({ name: r.name, kind: r.kind, file: r.file, start_line: r.start_line, end_line: r.end_line });
}
return out;
