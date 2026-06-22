var rows = Array.isArray(DATA) ? DATA : (DATA && DATA.results ? DATA.results : []);
return rows.map(function(r) {
  return { name: r.name, file: r.file, kind: r.kind };
});
