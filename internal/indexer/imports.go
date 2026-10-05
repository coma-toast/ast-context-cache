package indexer

import (
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	sitter "github.com/smacker/go-tree-sitter"
)

const (
	insertScopedImportEdgeQuery = "INSERT INTO edges (source_file, source_symbol, target, kind, project_path) VALUES (?, NULLIF(?, ''), ?, 'import', ?)"
	insertImportNamesEdgeQuery  = "INSERT INTO edges (source_file, source_symbol, target, kind, project_path) VALUES (?, NULLIF(?, ''), ?, 'import_names', ?)"
)

// ImportNamesSep joins an import_names edge's module and the names it binds:
// target "model_manager.mm_config::HOSTS,load_config".
const ImportNamesSep = "::"

// importRef is one import statement: the module it loads and, when the statement
// binds explicit names, those names. The enclosing function or method is kept so
// a function-local import records which caller depends on the module.
type importRef struct {
	Module string
	// Names are the names the statement binds from Module. "*" stands for the
	// whole module (star, namespace and default imports); nil means the statement
	// names nothing (import a.b, require, dynamic import) and so may use anything.
	Names []string
	// Scope is the dotted enclosing function/method/class ("Client.load_model"),
	// "" at module level.
	Scope string
}

// collectImports returns every import in a parsed file. Python, JS/TS and bash
// allow imports inside function bodies and blocks, so their whole tree is walked;
// Go and HCL only import at the top level.
func collectImports(root *sitter.Node, content []byte, lang string) []importRef {
	var out []importRef
	switch lang {
	case "python", "javascript", "typescript", "tsx", "bash":
		walkImports(root, content, lang, "", &out)
		return out
	}
	for _, node := range collectTopLevelNodes(root, lang) {
		for _, imp := range extractImports(node, content, lang) {
			out = append(out, importRef{Module: imp})
		}
	}
	return out
}

// insertImportEdges writes one 'import' edge per statement (target = module,
// source_symbol = enclosing function) plus, for statements that bind explicit
// names, one 'import_names' edge whose target is module::name1,name2. Keeping
// the per-statement pairing lets the impact graph tell `from m import a` (which
// cannot reach m.b) apart from `import m`.
func insertImportEdges(exec db.Execer, refs []importRef, filePath, projectPath string) error {
	for _, r := range refs {
		if r.Module == "" {
			continue
		}
		if _, err := exec.Exec(insertScopedImportEdgeQuery,
			filePath, r.Scope, r.Module, projectPath); err != nil {
			return err
		}
		if len(r.Names) == 0 {
			continue
		}
		if _, err := exec.Exec(insertImportNamesEdgeQuery,
			filePath, r.Scope, r.Module+ImportNamesSep+strings.Join(r.Names, ","), projectPath); err != nil {
			return err
		}
	}
	return nil
}

// importWalkSkip lists node types that can never contain an import, so the walk
// does not descend into the bulk of expression trees.
var importWalkSkip = map[string]map[string]bool{
	"python": {
		"expression_statement": true, "return_statement": true, "string": true, "comment": true,
		"decorator": true, "assert_statement": true, "raise_statement": true, "delete_statement": true,
		"global_statement": true, "nonlocal_statement": true, "pass_statement": true,
		"break_statement": true, "continue_statement": true, "parameters": true, "type": true,
	},
	"javascript": jsImportWalkSkip,
	"typescript": jsImportWalkSkip,
	"tsx":        jsImportWalkSkip,
	"bash":       {"comment": true, "heredoc_body": true, "raw_string": true},
}

var jsImportWalkSkip = map[string]bool{
	"comment": true, "template_string": true, "regex": true, "number": true,
	"jsx_text": true, "type_annotation": true, "type_alias_declaration": true, "interface_declaration": true,
}

func walkImports(n *sitter.Node, content []byte, lang, scope string, out *[]importRef) {
	if n == nil || n.IsNull() {
		return
	}
	nodeType := n.Type()
	if importWalkSkip[lang][nodeType] {
		return
	}
	switch lang {
	case "python":
		if refs, ok := pythonImport(n, content, scope); ok {
			*out = append(*out, refs...)
			return
		}
	case "javascript", "typescript", "tsx":
		if ref, ok := jsImport(n, content, scope); ok {
			if ref.Module != "" {
				*out = append(*out, ref)
			}
			if nodeType != "export_statement" && nodeType != "call_expression" {
				return
			}
		}
	case "bash":
		if nodeType == "command" {
			for _, imp := range extractImports(n, content, lang) {
				*out = append(*out, importRef{Module: imp, Scope: scope})
			}
		}
	}
	if name := scopeName(n, content, lang); name != "" {
		if scope != "" {
			name = scope + "." + name
		}
		scope = name
	}
	for i := 0; i < int(n.NamedChildCount()); i++ {
		walkImports(n.NamedChild(i), content, lang, scope, out)
	}
}

// scopeName names the function, method or class a node opens, or "".
func scopeName(n *sitter.Node, content []byte, lang string) string {
	switch lang {
	case "python":
		switch n.Type() {
		case "function_definition", "class_definition":
			return fieldContent(n, content, "name")
		}
	case "javascript", "typescript", "tsx":
		switch n.Type() {
		case "function_declaration", "generator_function_declaration", "class_declaration",
			"abstract_class_declaration", "method_definition", "function_expression", "function", "class":
			return fieldContent(n, content, "name")
		case "variable_declarator", "public_field_definition", "field_definition":
			value := n.ChildByFieldName("value")
			if value == nil {
				return ""
			}
			switch value.Type() {
			case "arrow_function", "function_expression", "function", "generator_function":
				if name := fieldContent(n, content, "name"); name != "" {
					return name
				}
				return fieldContent(n, content, "property")
			}
		}
	case "bash":
		if n.Type() == "function_definition" {
			return fieldContent(n, content, "name")
		}
	}
	return ""
}

// pythonImport handles `import a.b as c, d` and `from .m import x as y, z` / `*`.
func pythonImport(n *sitter.Node, content []byte, scope string) ([]importRef, bool) {
	switch n.Type() {
	case "import_statement":
		var refs []importRef
		for i := 0; i < int(n.NamedChildCount()); i++ {
			child := n.NamedChild(i)
			switch child.Type() {
			case "dotted_name":
				refs = append(refs, importRef{Module: child.Content(content), Scope: scope})
			case "aliased_import":
				if m := fieldContent(child, content, "name"); m != "" {
					refs = append(refs, importRef{Module: m, Scope: scope})
				}
			}
		}
		return refs, true
	case "import_from_statement":
		module := fieldContent(n, content, "module_name")
		if module == "" {
			return nil, true
		}
		var names []string
		for i := 0; i < int(n.ChildCount()); i++ {
			child := n.Child(i)
			if child.Type() == "wildcard_import" {
				names = append(names, "*")
				continue
			}
			if n.FieldNameForChild(i) != "name" {
				continue
			}
			switch child.Type() {
			case "dotted_name":
				names = append(names, child.Content(content))
			case "aliased_import":
				if name := fieldContent(child, content, "name"); name != "" {
					names = append(names, name)
				}
			}
		}
		return []importRef{{Module: module, Names: dedupeNames(names), Scope: scope}}, true
	case "future_import_statement":
		return nil, true
	}
	return nil, false
}

// jsImport handles static imports, re-exports (`export {a} from './m'`), CommonJS
// require('m') and dynamic import('m'). ok reports that n is one of those forms.
func jsImport(n *sitter.Node, content []byte, scope string) (importRef, bool) {
	switch n.Type() {
	case "import_statement":
		ref := importRef{Module: jsStringField(n, content, "source"), Scope: scope}
		for i := 0; i < int(n.NamedChildCount()); i++ {
			child := n.NamedChild(i)
			switch child.Type() {
			case "import_require_clause":
				// import x = require('m') binds the whole module.
				ref.Module = jsStringField(child, content, "source")
			case "import_clause":
				ref.Names = jsImportClauseNames(child, content)
			}
		}
		return ref, true
	case "export_statement":
		source := jsStringField(n, content, "source")
		if source == "" {
			return importRef{}, false
		}
		ref := importRef{Module: source, Scope: scope, Names: []string{"*"}}
		if clause := firstNamedChildOfType(n, "export_clause"); clause != nil {
			ref.Names = specifierNames(clause, content, "export_specifier")
		}
		return ref, true
	case "call_expression":
		fn := n.ChildByFieldName("function")
		if fn == nil {
			return importRef{}, false
		}
		if fn.Type() != "import" && !(fn.Type() == "identifier" && fn.Content(content) == "require") {
			return importRef{}, false
		}
		args := n.ChildByFieldName("arguments")
		if args == nil || args.NamedChildCount() == 0 {
			return importRef{}, true
		}
		arg := args.NamedChild(0)
		if arg.Type() != "string" {
			return importRef{}, true
		}
		return importRef{Module: trimJSString(arg.Content(content)), Scope: scope}, true
	}
	return importRef{}, false
}

// jsImportClauseNames lists what an import clause binds. Default and namespace
// imports expose the whole module (the default export's local name says nothing
// about which declaration it is), so they read as "*".
func jsImportClauseNames(clause *sitter.Node, content []byte) []string {
	var names []string
	for i := 0; i < int(clause.NamedChildCount()); i++ {
		child := clause.NamedChild(i)
		switch child.Type() {
		case "identifier", "namespace_import":
			names = append(names, "*")
		case "named_imports":
			names = append(names, specifierNames(child, content, "import_specifier")...)
		}
	}
	return dedupeNames(names)
}

func specifierNames(list *sitter.Node, content []byte, specType string) []string {
	var names []string
	for i := 0; i < int(list.NamedChildCount()); i++ {
		spec := list.NamedChild(i)
		if spec.Type() != specType {
			continue
		}
		if name := trimJSString(fieldContent(spec, content, "name")); name != "" {
			names = append(names, name)
		}
	}
	return dedupeNames(names)
}

func jsStringField(n *sitter.Node, content []byte, field string) string {
	s := n.ChildByFieldName(field)
	if s == nil {
		return ""
	}
	return trimJSString(s.Content(content))
}

func trimJSString(s string) string {
	return strings.Trim(s, "'\"`")
}

func fieldContent(n *sitter.Node, content []byte, field string) string {
	c := n.ChildByFieldName(field)
	if c == nil {
		return ""
	}
	return c.Content(content)
}

func firstNamedChildOfType(n *sitter.Node, typ string) *sitter.Node {
	for i := 0; i < int(n.NamedChildCount()); i++ {
		if c := n.NamedChild(i); c.Type() == typ {
			return c
		}
	}
	return nil
}

func dedupeNames(names []string) []string {
	if len(names) < 2 {
		return names
	}
	seen := make(map[string]bool, len(names))
	out := names[:0]
	for _, n := range names {
		if !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	return out
}
