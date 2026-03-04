package indexer

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/coma-toast/ast-context-cache/internal/db"
	sitter "github.com/smacker/go-tree-sitter"
)

type yamlSymbol struct {
	Name      string
	Kind      string
	StartLine int
	EndLine   int
}

func indexYAMLTree(root *sitter.Node, content []byte, lines []string, filePath, projectPath string) (int, error) {
	doc := findDocumentNode(root)
	if doc == nil {
		return 0, nil
	}

	var syms []yamlSymbol
	blockNode := findBlockNode(doc)
	if blockNode == nil {
		return 0, nil
	}

	switch blockNode.Type() {
	case "block_mapping":
		syms = extractMappingSymbols(blockNode, content)
	case "block_sequence":
		if isAnsiblePlaybook(blockNode, content) {
			syms = extractAnsiblePlaybook(blockNode, content)
		} else {
			syms = extractSequenceSymbols(blockNode, content)
		}
	}

	imports := extractYAMLImports(blockNode, content)
	for _, imp := range imports {
		db.DB.Exec("INSERT INTO edges (source_file, target, kind, project_path) VALUES (?, ?, 'import', ?)",
			filePath, imp, projectPath)
	}

	count := 0
	for _, sym := range syms {
		if sym.Name == "" {
			continue
		}
		code := ""
		if sym.StartLine > 0 && sym.StartLine <= len(lines) {
			code = strings.TrimSpace(lines[sym.StartLine-1])
		}
		fqn := fmt.Sprintf("%s.%s", filepath.Base(filePath), sym.Name)
		skeleton := ""
		if sym.StartLine > 0 && sym.EndLine <= len(lines) {
			src := strings.Join(lines[sym.StartLine-1:sym.EndLine], "\n")
			skeleton = ExtractSkeleton(src, "yaml", sym.Kind)
		}
		_, err := db.DB.Exec("INSERT INTO symbols (name, kind, file, start_line, end_line, code, fqn, project_path, skeleton) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
			sym.Name, sym.Kind, filePath, sym.StartLine, sym.EndLine, code, fqn, projectPath, skeleton)
		if err == nil {
			count++
		}
	}
	db.UpsertIndexedFile(filePath, projectPath, time.Now())
	return count, nil
}

func findDocumentNode(root *sitter.Node) *sitter.Node {
	for i := 0; i < int(root.NamedChildCount()); i++ {
		child := root.NamedChild(i)
		if child.Type() == "document" {
			return child
		}
	}
	// Handle ERROR root by walking its children directly
	if root.Type() == "ERROR" || root.Type() == "stream" {
		for i := 0; i < int(root.ChildCount()); i++ {
			child := root.Child(i)
			if child.Type() == "document" {
				return child
			}
			if child.Type() == "block_node" || child.Type() == "block_mapping" || child.Type() == "block_sequence" {
				return child
			}
		}
	}
	return root
}

func findBlockNode(node *sitter.Node) *sitter.Node {
	switch node.Type() {
	case "block_mapping", "block_sequence":
		return node
	}
	for i := 0; i < int(node.NamedChildCount()); i++ {
		child := node.NamedChild(i)
		switch child.Type() {
		case "block_mapping", "block_sequence":
			return child
		case "block_node":
			return findBlockNode(child)
		}
	}
	// Walk unnamed children as fallback (for ERROR nodes)
	for i := 0; i < int(node.ChildCount()); i++ {
		child := node.Child(i)
		switch child.Type() {
		case "block_mapping", "block_sequence":
			return child
		case "block_node", "block_sequence_item":
			if result := findBlockNode(child); result != nil {
				return result
			}
		}
	}
	return nil
}

// extractMappingSymbols extracts top-level mapping keys as symbols.
func extractMappingSymbols(mapping *sitter.Node, content []byte) []yamlSymbol {
	var syms []yamlSymbol
	for i := 0; i < int(mapping.NamedChildCount()); i++ {
		pair := mapping.NamedChild(i)
		if pair.Type() != "block_mapping_pair" {
			continue
		}
		key := yamlPairKey(pair, content)
		if key == "" {
			continue
		}
		syms = append(syms, yamlSymbol{
			Name:      key,
			Kind:      "key",
			StartLine: int(pair.StartPoint().Row) + 1,
			EndLine:   int(pair.EndPoint().Row) + 1,
		})
	}
	return syms
}

// extractSequenceSymbols extracts items from a non-Ansible top-level sequence.
func extractSequenceSymbols(seq *sitter.Node, content []byte) []yamlSymbol {
	var syms []yamlSymbol
	for i := 0; i < int(seq.NamedChildCount()); i++ {
		item := seq.NamedChild(i)
		if item.Type() != "block_sequence_item" {
			continue
		}
		name := fmt.Sprintf("item[%d]", i)
		mapping := findBlockNode(item)
		if mapping != nil && mapping.Type() == "block_mapping" {
			if n := yamlMappingValue(mapping, content, "name"); n != "" {
				name = n
			}
		}
		syms = append(syms, yamlSymbol{
			Name:      name,
			Kind:      "item",
			StartLine: int(item.StartPoint().Row) + 1,
			EndLine:   int(item.EndPoint().Row) + 1,
		})
	}
	return syms
}

func isAnsiblePlaybook(seq *sitter.Node, content []byte) bool {
	for i := 0; i < int(seq.NamedChildCount()); i++ {
		item := seq.NamedChild(i)
		if item.Type() != "block_sequence_item" {
			continue
		}
		mapping := findBlockNode(item)
		if mapping != nil && mapping.Type() == "block_mapping" {
			if hasYAMLKey(mapping, content, "hosts") || hasYAMLKey(mapping, content, "tasks") || hasYAMLKey(mapping, content, "roles") {
				return true
			}
		}
	}
	return false
}

func extractAnsiblePlaybook(seq *sitter.Node, content []byte) []yamlSymbol {
	var syms []yamlSymbol
	for i := 0; i < int(seq.NamedChildCount()); i++ {
		item := seq.NamedChild(i)
		if item.Type() != "block_sequence_item" {
			continue
		}
		mapping := findBlockNode(item)
		if mapping == nil || mapping.Type() != "block_mapping" {
			continue
		}
		playName := yamlMappingValue(mapping, content, "name")
		if playName == "" {
			playName = fmt.Sprintf("play[%d]", i)
		}
		syms = append(syms, yamlSymbol{
			Name:      playName,
			Kind:      "play",
			StartLine: int(item.StartPoint().Row) + 1,
			EndLine:   int(item.EndPoint().Row) + 1,
		})
		syms = append(syms, extractAnsibleTasks(mapping, content, "tasks", "task")...)
		syms = append(syms, extractAnsibleTasks(mapping, content, "handlers", "handler")...)
		syms = append(syms, extractAnsibleTasks(mapping, content, "pre_tasks", "task")...)
		syms = append(syms, extractAnsibleTasks(mapping, content, "post_tasks", "task")...)
	}
	return syms
}

func extractAnsibleTasks(mapping *sitter.Node, content []byte, sectionKey, kind string) []yamlSymbol {
	var syms []yamlSymbol
	for i := 0; i < int(mapping.NamedChildCount()); i++ {
		pair := mapping.NamedChild(i)
		if pair.Type() != "block_mapping_pair" {
			continue
		}
		if yamlPairKey(pair, content) != sectionKey {
			continue
		}
		taskSeq := findBlockNode(pair)
		if taskSeq == nil || taskSeq.Type() != "block_sequence" {
			continue
		}
		for j := 0; j < int(taskSeq.NamedChildCount()); j++ {
			taskItem := taskSeq.NamedChild(j)
			if taskItem.Type() != "block_sequence_item" {
				continue
			}
			taskMapping := findBlockNode(taskItem)
			if taskMapping == nil || taskMapping.Type() != "block_mapping" {
				continue
			}
			taskName := yamlMappingValue(taskMapping, content, "name")
			if taskName == "" {
				taskName = fmt.Sprintf("%s[%d]", sectionKey, j)
			}
			syms = append(syms, yamlSymbol{
				Name:      taskName,
				Kind:      kind,
				StartLine: int(taskItem.StartPoint().Row) + 1,
				EndLine:   int(taskItem.EndPoint().Row) + 1,
			})
		}
	}
	return syms
}

// yamlPairKey extracts the key string from a block_mapping_pair node.
func yamlPairKey(pair *sitter.Node, content []byte) string {
	for i := 0; i < int(pair.ChildCount()); i++ {
		child := pair.Child(i)
		if child.Type() == ":" {
			break
		}
		if child.Type() == "flow_node" {
			return extractScalarText(child, content)
		}
	}
	return ""
}

// yamlMappingValue finds a key in a block_mapping and returns its scalar value.
func yamlMappingValue(mapping *sitter.Node, content []byte, key string) string {
	for i := 0; i < int(mapping.NamedChildCount()); i++ {
		pair := mapping.NamedChild(i)
		if pair.Type() != "block_mapping_pair" {
			continue
		}
		if yamlPairKey(pair, content) != key {
			continue
		}
		for j := 0; j < int(pair.ChildCount()); j++ {
			child := pair.Child(j)
			if child.Type() == "flow_node" && j > 0 {
				return extractScalarText(child, content)
			}
		}
		// Value might be in a block_node
		for j := 0; j < int(pair.NamedChildCount()); j++ {
			child := pair.NamedChild(j)
			if child.Type() == "flow_node" {
				text := extractScalarText(child, content)
				if text != key {
					return text
				}
			}
		}
	}
	return ""
}

func hasYAMLKey(mapping *sitter.Node, content []byte, key string) bool {
	for i := 0; i < int(mapping.NamedChildCount()); i++ {
		pair := mapping.NamedChild(i)
		if pair.Type() == "block_mapping_pair" && yamlPairKey(pair, content) == key {
			return true
		}
	}
	return false
}

var ansibleImportKeys = map[string]bool{
	"import_playbook": true, "include_tasks": true, "import_tasks": true,
	"include_role": true, "import_role": true, "include": true,
}

func extractYAMLImports(root *sitter.Node, content []byte) []string {
	var imports []string
	walkYAMLMappings(root, content, func(key, value string) {
		if ansibleImportKeys[key] && value != "" {
			imports = append(imports, value)
		}
		if key == "roles" {
			return
		}
	})
	imports = append(imports, extractYAMLRoleImports(root, content)...)
	return imports
}

func walkYAMLMappings(node *sitter.Node, content []byte, fn func(key, value string)) {
	if node.Type() == "block_mapping_pair" {
		key := yamlPairKey(node, content)
		val := ""
		for i := 0; i < int(node.NamedChildCount()); i++ {
			child := node.NamedChild(i)
			if child.Type() == "flow_node" {
				text := extractScalarText(child, content)
				if text != key {
					val = text
					break
				}
			}
		}
		fn(key, val)
	}
	for i := 0; i < int(node.NamedChildCount()); i++ {
		walkYAMLMappings(node.NamedChild(i), content, fn)
	}
}

func extractYAMLRoleImports(root *sitter.Node, content []byte) []string {
	var imports []string
	walkYAMLMappings(root, content, func(key, value string) {
		if key == "role" && value != "" {
			imports = append(imports, value)
		}
	})
	return imports
}

func extractScalarText(node *sitter.Node, content []byte) string {
	for i := 0; i < int(node.NamedChildCount()); i++ {
		child := node.NamedChild(i)
		switch child.Type() {
		case "plain_scalar":
			return extractScalarText(child, content)
		case "string_scalar", "boolean_scalar", "integer_scalar", "float_scalar", "null_scalar":
			return child.Content(content)
		case "double_quote_scalar", "single_quote_scalar":
			text := child.Content(content)
			if len(text) >= 2 {
				return text[1 : len(text)-1]
			}
			return text
		}
	}
	return node.Content(content)
}
