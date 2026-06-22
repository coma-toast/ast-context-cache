var rows = Array.isArray(DATA) ? DATA : (DATA && DATA.results ? DATA.results : []);
return rows.slice(0, 20).map(function(r) {
  return { name: r.name, kind: r.kind, file: r.file };
});
