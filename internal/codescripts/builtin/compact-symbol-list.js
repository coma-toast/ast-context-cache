var rows = Array.isArray(DATA) ? DATA : (DATA && DATA.results ? DATA.results : []);
return rows.map(function(r) {
  return {
    name: r.name,
    kind: r.kind,
    file: r.file,
    start_line: r.start_line,
    end_line: r.end_line
  };
});
