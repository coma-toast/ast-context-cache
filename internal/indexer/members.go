package indexer

import (
	"strings"

	"github.com/coma-toast/ast-context-cache/internal/db"
	sitter "github.com/smacker/go-tree-sitter"
)

func init() { db.ParserVersion = ParserVersion }

// symbolNode is one declaration found in a parsed file: what it is, the node it
// spans, and — for a class member or Go method — the enclosing type it belongs
// to, so methods are indexed as their own symbols instead of only being inlined
// into their class's skeleton.
type symbolNode struct {
	SymbolDef
	Node   *sitter.Node
	Parent string
}

// Qualified returns Parent.Name for a member and Name for a top-level symbol.
func (s symbolNode) Qualified() string {
	if s.Parent == "" {
		return s.Name
	}
	return s.Parent + "." + s.Name
}

// declaredSymbols returns the symbol declared by a top-level node followed by
// every method (and nested class) inside it when it is a class.
func declaredSymbols(node *sitter.Node, content []byte, lang string) []symbolNode {
	sym := extractSymbol(node, content, lang)
	if sym == nil || sym.Name == "" {
		return nil
	}
	out := []symbolNode{{SymbolDef: *sym, Node: node}}
	switch {
	case lang == "go" && sym.Kind == "method":
		out[0].Parent = goReceiverType(node, content)
	case sym.Kind == "class":
		out = append(out, classMembers(unwrapDeclaration(node, lang), content, lang, sym.Name)...)
	}
	return out
}

// unwrapDeclaration returns the class/function node inside a Python decorator or
// a JS/TS export wrapper, or node itself when it is not wrapped.
func unwrapDeclaration(node *sitter.Node, lang string) *sitter.Node {
	switch node.Type() {
	case "decorated_definition":
		if def := node.ChildByFieldName("definition"); def != nil {
			return def
		}
	case "export_statement":
		if decl := node.ChildByFieldName("declaration"); decl != nil {
			return decl
		}
		for i := 0; i < int(node.NamedChildCount()); i++ {
			if c := node.NamedChild(i); c.Type() == "class_declaration" || c.Type() == "abstract_class_declaration" {
				return c
			}
		}
	}
	return node
}

// classMembers returns the methods and nested classes declared directly in a
// class body, each qualified by the enclosing class chain (Outer.Inner).
func classMembers(class *sitter.Node, content []byte, lang, parent string) []symbolNode {
	body := class.ChildByFieldName("body")
	if body == nil {
		return nil
	}
	var out []symbolNode
	for i := 0; i < int(body.NamedChildCount()); i++ {
		member := body.NamedChild(i)
		switch lang {
		case "python":
			def := unwrapDeclaration(member, lang)
			switch def.Type() {
			case "function_definition":
				if name := nodeName(def, content); name != "" {
					out = append(out, symbolNode{SymbolDef{name, "method"}, member, parent})
				}
			case "class_definition":
				if name := nodeName(def, content); name != "" {
					out = append(out, symbolNode{SymbolDef{name, "class"}, member, parent})
					out = append(out, classMembers(def, content, lang, parent+"."+name)...)
				}
			}
		case "javascript", "typescript", "tsx":
			switch member.Type() {
			case "method_definition", "abstract_method_signature":
				if name := nodeName(member, content); name != "" {
					out = append(out, symbolNode{SymbolDef{name, "method"}, member, parent})
				}
			}
		}
	}
	return out
}

// nodeName returns the identifier in a declaration's name field. Computed or
// string-literal member names ([Symbol.iterator], 'quoted'()) are skipped.
func nodeName(node *sitter.Node, content []byte) string {
	name := node.ChildByFieldName("name")
	if name == nil {
		return ""
	}
	switch name.Type() {
	case "identifier", "type_identifier", "property_identifier", "private_property_identifier", "field_identifier":
		return name.Content(content)
	}
	return ""
}

// goReceiverType returns the receiver's base type name of a Go method, without
// pointer or type arguments: (s *Server[T]) -> Server.
func goReceiverType(method *sitter.Node, content []byte) string {
	recv := method.ChildByFieldName("receiver")
	if recv == nil {
		return ""
	}
	var find func(n *sitter.Node) string
	find = func(n *sitter.Node) string {
		if n.Type() == "type_identifier" {
			return n.Content(content)
		}
		for i := 0; i < int(n.NamedChildCount()); i++ {
			if t := find(n.NamedChild(i)); t != "" {
				return t
			}
		}
		return ""
	}
	for i := 0; i < int(recv.NamedChildCount()); i++ {
		if param := recv.NamedChild(i); param.Type() == "parameter_declaration" {
			if typ := param.ChildByFieldName("type"); typ != nil {
				return find(typ)
			}
		}
	}
	return ""
}

// nodeSource returns exactly the text a node spans, dedented by its start column
// so an indented method (or one sharing a line with its class) reads as it would
// at top level.
func nodeSource(lines []string, start, end sitter.Point) string {
	var out []string
	col := int(start.Column)
	for r := int(start.Row); r <= int(end.Row) && r < len(lines); r++ {
		line := lines[r]
		if r == int(end.Row) && int(end.Column) <= len(line) {
			line = line[:end.Column]
		}
		if r == int(start.Row) {
			if col <= len(line) {
				line = line[col:]
			}
		} else {
			line = trimIndent(line, col)
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

// trimIndent strips up to n bytes of leading whitespace.
func trimIndent(line string, n int) string {
	i := 0
	for i < n && i < len(line) && (line[i] == ' ' || line[i] == '\t') {
		i++
	}
	return line[i:]
}

// symbolParserVersion is bumped whenever what tree-sitter languages extract per
// file changes, so files indexed by an older parser are re-indexed by the next
// catch-up instead of waiting for their mtime to change.
//
//	2: class methods (Python, JS/TS) and nested Python classes are symbols; TS
//	   class/interface/type names are read from the name field (type_identifier);
//	   Go method fqns carry the receiver type; multi-line signatures are kept
//	   whole in skeletons.
const symbolParserVersion = 2

// ParserVersion returns the parser version a file is indexed with today. Only
// languages whose extraction changed carry a version, so a bump re-indexes (and
// re-embeds) just those files.
func ParserVersion(file string) int {
	switch GetLanguage(file) {
	case "python", "javascript", "typescript", "tsx", "go":
		return symbolParserVersion
	}
	return 0
}
