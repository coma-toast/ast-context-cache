package indexer

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/bash"
	"github.com/smacker/go-tree-sitter/golang"
	"github.com/smacker/go-tree-sitter/hcl"
	"github.com/smacker/go-tree-sitter/javascript"
	"github.com/smacker/go-tree-sitter/python"
	"github.com/smacker/go-tree-sitter/typescript/tsx"
	"github.com/smacker/go-tree-sitter/typescript/typescript"
	"github.com/smacker/go-tree-sitter/yaml"
)

var SkipDirs = map[string]bool{
	"node_modules": true, "vendor": true, "venv": true, "env": true,
	"__pycache__": true, "dist": true, "build": true, ".astcache": true,
	"target": true, "bower_components": true, "third_party": true,
	"site-packages": true, "coverage": true, "tmp": true,
	"pkg": true, "Pods": true,
}

func ShouldSkipDir(name string) bool {
	return strings.HasPrefix(name, ".") || SkipDirs[name]
}

func GetLanguage(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".py":
		return "python"
	case ".js", ".jsx":
		return "javascript"
	case ".ts":
		return "typescript"
	case ".tsx":
		return "tsx"
	case ".go":
		return "go"
	case ".sh":
		return "bash"
	case ".fish":
		return "fish"
	case ".yaml", ".yml":
		return "yaml"
	case ".tf", ".tfvars":
		return "hcl"
	}
	return ""
}

func IsCodeFile(path string) bool {
	return GetLanguage(path) != ""
}

func getSitterLanguage(lang string) *sitter.Language {
	switch lang {
	case "python":
		return python.GetLanguage()
	case "javascript":
		return javascript.GetLanguage()
	case "typescript":
		return typescript.GetLanguage()
	case "tsx":
		return tsx.GetLanguage()
	case "go":
		return golang.GetLanguage()
	case "bash":
		return bash.GetLanguage()
	case "yaml":
		return yaml.GetLanguage()
	case "hcl":
		return hcl.GetLanguage()
	}
	return nil
}

type SymbolDef struct {
	Name string
	Kind string
}

func extractSymbol(node *sitter.Node, content []byte, lang string) *SymbolDef {
	nodeType := node.Type()

	switch lang {
	case "python":
		switch nodeType {
		case "function_definition":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "function"}
		case "class_definition":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "class"}
		case "decorated_definition":
			for j := 0; j < int(node.NamedChildCount()); j++ {
				if s := extractSymbol(node.NamedChild(j), content, lang); s != nil {
					return s
				}
			}
		}

	case "javascript":
		switch nodeType {
		case "function_declaration":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "function"}
		case "class_declaration":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "class"}
		case "lexical_declaration", "variable_declaration":
			if name := getVarDeclName(node, content); name != "" {
				return &SymbolDef{name, "variable"}
			}
		case "export_statement":
			for j := 0; j < int(node.NamedChildCount()); j++ {
				if s := extractSymbol(node.NamedChild(j), content, lang); s != nil {
					return s
				}
			}
		}

	case "typescript", "tsx":
		switch nodeType {
		case "function_declaration":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "function"}
		case "class_declaration":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "class"}
		case "interface_declaration":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "interface"}
		case "type_alias_declaration":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "type"}
		case "enum_declaration":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "enum"}
		case "lexical_declaration", "variable_declaration":
			if name := getVarDeclName(node, content); name != "" {
				return &SymbolDef{name, "variable"}
			}
		case "export_statement":
			for j := 0; j < int(node.NamedChildCount()); j++ {
				if s := extractSymbol(node.NamedChild(j), content, lang); s != nil {
					return s
				}
			}
		}

	case "go":
		switch nodeType {
		case "function_declaration":
			return &SymbolDef{getFirstChildByType(node, content, "identifier"), "function"}
		case "method_declaration":
			return &SymbolDef{getFirstChildByType(node, content, "field_identifier"), "method"}
		case "type_declaration":
			for j := 0; j < int(node.NamedChildCount()); j++ {
				child := node.NamedChild(j)
				if child.Type() == "type_spec" {
					name := getFirstChildByType(child, content, "type_identifier")
					if name != "" {
						kind := "type"
						for k := 0; k < int(child.NamedChildCount()); k++ {
							switch child.NamedChild(k).Type() {
							case "struct_type":
								kind = "struct"
							case "interface_type":
								kind = "interface"
							}
						}
						return &SymbolDef{name, kind}
					}
				}
			}
		}

	case "bash":
		if nodeType == "function_definition" {
			return &SymbolDef{getFirstChildByType(node, content, "word"), "function"}
		}

	case "hcl":
		switch nodeType {
		case "block":
			return extractHCLBlock(node, content)
		case "attribute":
			name := getFirstChildByType(node, content, "identifier")
			if name != "" {
				return &SymbolDef{name, "variable"}
			}
		}
	}

	return nil
}

func extractHCLBlock(node *sitter.Node, content []byte) *SymbolDef {
	var blockType string
	var labels []string
	for i := 0; i < int(node.NamedChildCount()); i++ {
		child := node.NamedChild(i)
		switch child.Type() {
		case "identifier":
			if blockType == "" {
				blockType = child.Content(content)
			}
		case "string_lit":
			if lit := getFirstChildByType(child, content, "template_literal"); lit != "" {
				labels = append(labels, lit)
			}
		}
	}
	if blockType == "" {
		return nil
	}
	name := blockType
	if len(labels) > 0 {
		name = strings.Join(labels, ".")
	}
	return &SymbolDef{name, blockType}
}

func extractImports(node *sitter.Node, content []byte, lang string) []string {
	var imports []string
	nodeType := node.Type()

	switch lang {
	case "python":
		switch nodeType {
		case "import_statement":
			for i := 0; i < int(node.NamedChildCount()); i++ {
				child := node.NamedChild(i)
				if child.Type() == "dotted_name" || child.Type() == "aliased_import" {
					imports = append(imports, child.Content(content))
				}
			}
		case "import_from_statement":
			for i := 0; i < int(node.NamedChildCount()); i++ {
				child := node.NamedChild(i)
				if child.Type() == "dotted_name" || child.Type() == "relative_import" {
					imports = append(imports, child.Content(content))
					break
				}
			}
		}

	case "javascript", "typescript", "tsx":
		if nodeType == "import_statement" {
			for i := 0; i < int(node.NamedChildCount()); i++ {
				child := node.NamedChild(i)
				if child.Type() == "string" || child.Type() == "string_fragment" {
					src := strings.Trim(child.Content(content), "'\"")
					if src != "" {
						imports = append(imports, src)
					}
				}
			}
		}

	case "go":
		if nodeType == "import_declaration" {
			for i := 0; i < int(node.NamedChildCount()); i++ {
				child := node.NamedChild(i)
				switch child.Type() {
				case "import_spec":
					p := getFirstChildByType(child, content, "interpreted_string_literal")
					if p != "" {
						imports = append(imports, strings.Trim(p, "\""))
					}
				case "import_spec_list":
					for j := 0; j < int(child.NamedChildCount()); j++ {
						spec := child.NamedChild(j)
						if spec.Type() == "import_spec" {
							p := getFirstChildByType(spec, content, "interpreted_string_literal")
							if p != "" {
								imports = append(imports, strings.Trim(p, "\""))
							}
						}
					}
				}
			}
		}

	case "bash":
		if nodeType == "command" {
			name := getFirstChildByType(node, content, "command_name")
			if name == "source" || name == "." {
				for i := 0; i < int(node.NamedChildCount()); i++ {
					child := node.NamedChild(i)
					if child.Type() == "word" || child.Type() == "string" {
						val := strings.Trim(child.Content(content), "'\"")
						if val != name {
							imports = append(imports, val)
						}
					}
				}
			}
		}

	case "hcl":
		if nodeType == "block" {
			blockType := getFirstChildByType(node, content, "identifier")
			if blockType == "module" {
				if src := hclBlockAttrValue(node, content, "source"); src != "" {
					imports = append(imports, src)
				}
			}
		}
	}

	return imports
}

func hclBlockAttrValue(block *sitter.Node, content []byte, attrName string) string {
	for i := 0; i < int(block.NamedChildCount()); i++ {
		child := block.NamedChild(i)
		if child.Type() == "body" {
			for j := 0; j < int(child.NamedChildCount()); j++ {
				attr := child.NamedChild(j)
				if attr.Type() != "attribute" {
					continue
				}
				name := getFirstChildByType(attr, content, "identifier")
				if name != attrName {
					continue
				}
				for k := 0; k < int(attr.NamedChildCount()); k++ {
					expr := attr.NamedChild(k)
					if expr.Type() == "expression" {
						return extractHCLStringValue(expr, content)
					}
				}
			}
		}
	}
	return ""
}

func extractHCLStringValue(expr *sitter.Node, content []byte) string {
	for i := 0; i < int(expr.NamedChildCount()); i++ {
		child := expr.NamedChild(i)
		if child.Type() == "literal_value" {
			for j := 0; j < int(child.NamedChildCount()); j++ {
				lit := child.NamedChild(j)
				if lit.Type() == "string_lit" {
					if tpl := getFirstChildByType(lit, content, "template_literal"); tpl != "" {
						return tpl
					}
				}
			}
		}
	}
	return ""
}

func getFirstChildByType(node *sitter.Node, content []byte, nodeType string) string {
	for i := 0; i < int(node.NamedChildCount()); i++ {
		child := node.NamedChild(i)
		if child.Type() == nodeType {
			return child.Content(content)
		}
	}
	return ""
}

func getVarDeclName(node *sitter.Node, content []byte) string {
	for i := 0; i < int(node.NamedChildCount()); i++ {
		child := node.NamedChild(i)
		if child.Type() == "variable_declarator" {
			return getFirstChildByType(child, content, "identifier")
		}
	}
	return ""
}

func IndexFile(filePath, projectPath string) (int, error) {
	lang := GetLanguage(filePath)
	if lang == "" {
		return 0, fmt.Errorf("unsupported: %s", filePath)
	}
	if lang == "fish" {
		return IndexFishFile(filePath, projectPath)
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		return 0, err
	}

	sitterLang := getSitterLanguage(lang)
	if sitterLang == nil {
		return 0, fmt.Errorf("no parser for: %s", lang)
	}

	parser := sitter.NewParser()
	parser.SetLanguage(sitterLang)

	tree, err := parser.ParseCtx(context.Background(), nil, content)
	if err != nil {
		return 0, err
	}
	defer tree.Close()

	db.DB.Exec("DELETE FROM symbols WHERE file = ? AND project_path = ?", filePath, projectPath)
	db.DB.Exec("DELETE FROM edges WHERE source_file = ? AND project_path = ?", filePath, projectPath)

	count := 0
	lines := strings.Split(string(content), "\n")
	root := tree.RootNode()

	if lang == "yaml" {
		return indexYAMLTree(root, content, lines, filePath, projectPath)
	}

	walkNodes := collectTopLevelNodes(root, lang)
	for _, node := range walkNodes {
		for _, imp := range extractImports(node, content, lang) {
			db.DB.Exec("INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, ?, 'import', ?)",
				filePath, imp, projectPath)
		}
		sym := extractSymbol(node, content, lang)
		if sym == nil || sym.Name == "" {
			continue
		}
		start := node.StartPoint()
		end := node.EndPoint()
		code := ""
		if int(start.Row) < len(lines) {
			code = strings.TrimSpace(lines[start.Row])
		}
		fqn := fmt.Sprintf("%s.%s", filepath.Base(filePath), sym.Name)
		skeleton := ""
		if int(start.Row) < len(lines) && int(end.Row) < len(lines) {
			src := strings.Join(lines[start.Row:end.Row+1], "\n")
			skeleton = ExtractSkeleton(src, lang, sym.Kind)
		}
		_, err := db.DB.Exec("INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path, skeleton) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
			sym.Name, sym.Kind, filePath, start.Row+1, end.Row+1, code, fqn, projectPath, skeleton)
		if err == nil {
			count++
		}
	}
	db.UpsertIndexedFile(filePath, projectPath, time.Now())
	return count, nil
}

// collectTopLevelNodes returns the nodes to walk for symbol extraction.
// For HCL, we descend into the body node since blocks are children of body, not config_file directly.
func collectTopLevelNodes(root *sitter.Node, lang string) []*sitter.Node {
	var nodes []*sitter.Node
	if lang == "hcl" {
		for i := 0; i < int(root.NamedChildCount()); i++ {
			child := root.NamedChild(i)
			if child.Type() == "body" {
				for j := 0; j < int(child.NamedChildCount()); j++ {
					nodes = append(nodes, child.NamedChild(j))
				}
				return nodes
			}
		}
	}
	for i := 0; i < int(root.NamedChildCount()); i++ {
		nodes = append(nodes, root.NamedChild(i))
	}
	return nodes
}

func IndexDirectory(dirPath, projectPath string) (int, error) {
	count := 0
	err := filepath.Walk(dirPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			if ShouldSkipDir(info.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if !IsCodeFile(path) {
			return nil
		}
		n, err := IndexFile(path, projectPath)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
		}
		count += n
		return nil
	})
	return count, err
}

func GetIndexStats(projectPath string) (map[string]interface{}, error) {
	var nodes, files int
	var err error
	if projectPath != "" {
		err = db.DB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols WHERE project_path = ?", projectPath).Scan(&nodes, &files)
	} else {
		err = db.DB.QueryRow("SELECT COUNT(*), COUNT(DISTINCT file) FROM symbols").Scan(&nodes, &files)
	}
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{"total_nodes": nodes, "total_files": files}, nil
}

func ReadSourceRange(file string, startLine, endLine int, cache map[string][]string) string {
	if _, ok := cache[file]; !ok {
		content, err := os.ReadFile(file)
		if err != nil {
			return ""
		}
		cache[file] = strings.Split(string(content), "\n")
	}
	lines := cache[file]
	if startLine < 1 || endLine > len(lines) || startLine > endLine {
		return ""
	}
	return strings.Join(lines[startLine-1:endLine], "\n")
}
