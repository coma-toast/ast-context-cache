package installer

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"

	"github.com/tailscale/hujson"

	"github.com/coma-toast/ast-context-cache/internal/errs"
)

// jsonDoc is a JSON or JSONC file edited in place through the hujson syntax tree. Edits splice
// new members into the tree and the file is re-packed as-is: Format and Standardize are never
// called, so comments, trailing commas, key order, and the user's whitespace survive (IN-3).
// Inserted values copy the file's indentation unit and newline style.
type jsonDoc struct {
	root      hujson.Value
	bom       bool
	nl        string
	unit      string
	multiline bool
}

// jsonNode is a value in the tree plus the indentation of the line it starts on, which is what
// a nested insert indents relative to.
type jsonNode struct {
	v      *hujson.Value
	indent string
}

// jsonObj is an object whose keys marshal in the order given, so inserted entries read the way
// host docs show them ("type" before "url") instead of alphabetically.
type jsonObj []jsonField

type jsonField struct {
	Key   string
	Value any
}

var utf8BOM = []byte{0xEF, 0xBB, 0xBF}

// parseJSONDoc parses data, which may be empty (a new file). A parse error aborts the edit: the
// caller must not write anything (IN-3).
func parseJSONDoc(data []byte) (*jsonDoc, error) {
	d := &jsonDoc{nl: "\n", unit: "  ", multiline: true}
	if bytes.HasPrefix(data, utf8BOM) {
		d.bom, data = true, data[len(utf8BOM):]
	}
	if bytes.Contains(data, []byte("\r\n")) {
		d.nl = "\r\n"
	}
	if len(bytes.TrimSpace(data)) == 0 {
		data = []byte("{}" + d.nl)
	}
	v, err := hujson.Parse(bytes.Clone(data))
	if err != nil {
		return nil, errs.WrapCodeMessage(errs.CodeInvalidInput, "failed to parse JSON", err)
	}
	obj, ok := v.Value.(*hujson.Object)
	if !ok {
		return nil, errs.NewCode(errs.CodeInvalidInput, "top-level JSON value is not an object")
	}
	d.root = v
	if ind, ml, found := leadLayout(objectLeads(obj)); found {
		d.multiline = ml
		if ml && ind != "" {
			d.unit = ind
		}
	}
	return d, nil
}

// MarshalJSON writes the fields in order.
func (o jsonObj) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, f := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		k, err := marshalNoEscape(f.Key)
		if err != nil {
			return nil, err
		}
		v, err := marshalNoEscape(f.Value)
		if err != nil {
			return nil, err
		}
		b.Write(k)
		b.WriteByte(':')
		b.Write(v)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// bytes re-packs the document, restoring a leading BOM.
func (d *jsonDoc) bytes() []byte {
	out := d.root.Pack()
	if d.bom {
		out = append(bytes.Clone(utf8BOM), out...)
	}
	return out
}

// find walks object members along path.
func (d *jsonDoc) find(path ...string) (jsonNode, bool) {
	n := jsonNode{v: &d.root}
	for _, key := range path {
		obj, ok := n.v.Value.(*hujson.Object)
		if !ok {
			return jsonNode{}, false
		}
		i := memberIndex(obj, key)
		if i < 0 {
			return jsonNode{}, false
		}
		n = jsonNode{v: &obj.Members[i].Value, indent: lineIndentOr(obj.Members[i].Name.BeforeExtra, n.indent)}
	}
	return n, true
}

// standard returns the canonical JSON of the value at path (comments stripped, keys sorted), for
// comparing and hashing entries independent of formatting.
func (d *jsonDoc) standard(path ...string) ([]byte, bool, error) {
	n, ok := d.find(path...)
	if !ok {
		return nil, false, nil
	}
	b, err := standardJSON(*n.v)
	return b, true, err
}

// set puts val at path, creating missing parent objects. It reports whether it had to create
// path[0], so uninstall can later remove a container the installer added.
func (d *jsonDoc) set(path []string, val any) (bool, error) {
	n := jsonNode{v: &d.root}
	for i, key := range path {
		obj, ok := n.v.Value.(*hujson.Object)
		if !ok {
			return false, errs.NewCode(errs.CodeInvalidInput, "JSON value is not an object", "key", strings.Join(path[:i], "."))
		}
		idx := memberIndex(obj, key)
		if idx == -2 {
			return false, errs.NewCode(errs.CodeConflict, "duplicate JSON key", "key", strings.Join(path[:i+1], "."))
		}
		if idx < 0 {
			v := val
			for j := len(path) - 1; j > i; j-- {
				v = jsonObj{{path[j], v}}
			}
			return i == 0 && len(path) > 1, d.appendMember(obj, n.indent, key, v)
		}
		m := &obj.Members[idx]
		ind := lineIndentOr(m.Name.BeforeExtra, n.indent)
		if i < len(path)-1 {
			n = jsonNode{v: &m.Value, indent: ind}
			continue
		}
		nv, err := d.render(val, ind, d.multiline)
		if err != nil {
			return false, err
		}
		m.Value.Value = nv.Value
	}
	return false, nil
}

// remove deletes the member at path. It reports whether the member existed.
func (d *jsonDoc) remove(path ...string) (bool, error) {
	parent, ok := d.find(path[:len(path)-1]...)
	if !ok {
		return false, nil
	}
	obj, ok := parent.v.Value.(*hujson.Object)
	if !ok {
		return false, nil
	}
	idx := memberIndex(obj, path[len(path)-1])
	if idx == -2 {
		return false, errs.NewCode(errs.CodeConflict, "duplicate JSON key", "key", strings.Join(path, "."))
	}
	if idx < 0 {
		return false, nil
	}
	removeMember(obj, idx)
	return true, nil
}

// appendElem appends val to the array at node.
func (d *jsonDoc) appendElem(n jsonNode, val any) error {
	arr, ok := n.v.Value.(*hujson.Array)
	if !ok {
		return errs.NewCode(errs.CodeInvalidInput, "JSON value is not an array")
	}
	leads := make([][]byte, len(arr.Elements))
	for i := range arr.Elements {
		leads[i] = arr.Elements[i].BeforeExtra
	}
	indent, ml := d.childLayout(leads, n.indent)
	v, err := d.render(val, indent, ml)
	if err != nil {
		return err
	}
	inline := firstLead(leads)
	v.BeforeExtra, arr.AfterExtra = d.appendLayout(arr.AfterExtra, indent, n.indent, ml, inline)
	if k := len(arr.Elements); k > 0 && arr.Elements[k-1].AfterExtra != nil {
		v.AfterExtra = []byte{}
	}
	arr.Elements = append(arr.Elements, v)
	return nil
}

func (d *jsonDoc) appendMember(obj *hujson.Object, ownIndent, key string, val any) error {
	leads := objectLeads(obj)
	indent, ml := d.childLayout(leads, ownIndent)
	v, err := d.render(val, indent, ml)
	if err != nil {
		return err
	}
	m := hujson.ObjectMember{Name: hujson.Value{Value: hujson.String(key)}, Value: v}
	m.Name.BeforeExtra, obj.AfterExtra = d.appendLayout(obj.AfterExtra, indent, ownIndent, ml, firstLead(leads))
	m.Value.BeforeExtra = []byte(" ")
	if !ml {
		m.Value.BeforeExtra = nil
	}
	if k := len(obj.Members); k > 0 {
		m.Name.AfterExtra = bytes.Clone(obj.Members[0].Name.AfterExtra)
		m.Value.BeforeExtra = bytes.Clone(obj.Members[0].Value.BeforeExtra)
		if obj.Members[k-1].Value.AfterExtra != nil {
			m.Value.AfterExtra = []byte{}
		}
	}
	obj.Members = append(obj.Members, m)
	return nil
}

// childLayout picks the indentation and line style for a new child of a composite whose
// existing children have the given leading extras.
func (d *jsonDoc) childLayout(leads [][]byte, ownIndent string) (string, bool) {
	if ind, ml, found := leadLayout(leads); found {
		if ml {
			return ind, true
		}
		return ownIndent, false
	}
	return ownIndent + d.unit, d.multiline
}

// appendLayout returns the leading extra for a child appended to a composite and the
// composite's new closing extra. Comments sitting before the closing bracket stay ahead of the
// new child, and the closing bracket keeps its own indentation.
func (d *jsonDoc) appendLayout(closing []byte, indent, ownIndent string, multiline bool, inlineLead []byte) ([]byte, []byte) {
	if !multiline {
		return bytes.Clone(inlineLead), closing
	}
	i := bytes.LastIndexByte(closing, '\n')
	if i < 0 {
		return []byte(string(closing) + d.nl + indent), []byte(d.nl + ownIndent)
	}
	head := bytes.TrimRight(closing[:i], "\r")
	return []byte(string(head) + d.nl + indent), []byte(d.nl + string(closing[i+1:]))
}

// render marshals val as a hujson value laid out for a line indented by indent.
func (d *jsonDoc) render(val any, indent string, multiline bool) (hujson.Value, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if multiline {
		enc.SetIndent(indent, d.unit)
	}
	if err := enc.Encode(val); err != nil {
		return hujson.Value{}, errs.WrapMessage("failed to encode JSON value", err)
	}
	s := strings.TrimRight(buf.String(), "\n")
	if d.nl != "\n" {
		s = strings.ReplaceAll(s, "\n", d.nl)
	}
	v, err := hujson.Parse([]byte(s))
	if err != nil {
		return hujson.Value{}, errs.WrapMessage("failed to parse rendered JSON value", err)
	}
	return v, nil
}

// removeMember deletes obj.Members[idx], keeping the object's trailing-comma style and any
// comments that sat above the removed member.
func removeMember(obj *hujson.Object, idx int) {
	m := obj.Members[idx]
	if idx < len(obj.Members)-1 {
		next := &obj.Members[idx+1].Name
		if hasComment(m.Name.BeforeExtra) {
			next.BeforeExtra = mergeExtra(m.Name.BeforeExtra, next.BeforeExtra)
		}
		obj.Members = append(obj.Members[:idx], obj.Members[idx+1:]...)
		return
	}
	obj.Members = obj.Members[:idx]
	obj.AfterExtra = closeAfterRemoval(m.Name.BeforeExtra, m.Value.AfterExtra != nil, lastTail(obj), obj.AfterExtra)
}

// removeElem deletes arr.Elements[idx] with the same rules as removeMember.
func removeElem(arr *hujson.Array, idx int) {
	e := arr.Elements[idx]
	if idx < len(arr.Elements)-1 {
		next := &arr.Elements[idx+1]
		if hasComment(e.BeforeExtra) {
			next.BeforeExtra = mergeExtra(e.BeforeExtra, next.BeforeExtra)
		}
		arr.Elements = append(arr.Elements[:idx], arr.Elements[idx+1:]...)
		return
	}
	arr.Elements = arr.Elements[:idx]
	var tail *hujson.Extra
	if k := len(arr.Elements); k > 0 {
		tail = &arr.Elements[k-1].AfterExtra
	}
	arr.AfterExtra = closeAfterRemoval(e.BeforeExtra, e.AfterExtra != nil, tail, arr.AfterExtra)
}

// closeAfterRemoval fixes up the composite after its last child was removed: the new last child
// takes over the trailing comma (or loses one), and the closing extra keeps removed comments.
// An emptied composite whose closing extra is only whitespace collapses to {} or [].
func closeAfterRemoval(removedLead []byte, hadTrailingComma bool, tail *hujson.Extra, closing []byte) []byte {
	if tail != nil {
		switch {
		case hadTrailingComma && *tail == nil:
			*tail = []byte{}
		case !hadTrailingComma && *tail != nil:
			closing = append(bytes.Clone(*tail), closing...)
			*tail = nil
		}
	}
	if hasComment(removedLead) {
		closing = mergeExtra(removedLead, closing)
	}
	if tail == nil && len(bytes.TrimSpace(closing)) == 0 {
		return nil
	}
	return closing
}

func lastTail(obj *hujson.Object) *hujson.Extra {
	if k := len(obj.Members); k > 0 {
		return &obj.Members[k-1].Value.AfterExtra
	}
	return nil
}

// memberIndex returns the index of key in obj, -1 when absent, or -2 when it appears twice.
func memberIndex(obj *hujson.Object, key string) int {
	idx := -1
	for i := range obj.Members {
		lit, ok := obj.Members[i].Name.Value.(hujson.Literal)
		if !ok || lit.String() != key {
			continue
		}
		if idx >= 0 {
			return -2
		}
		idx = i
	}
	return idx
}

func objectLeads(obj *hujson.Object) [][]byte {
	out := make([][]byte, len(obj.Members))
	for i := range obj.Members {
		out[i] = obj.Members[i].Name.BeforeExtra
	}
	return out
}

// leadLayout inspects the children's leading extras: the first that starts on a new line gives
// the indentation. found is false for a composite with no children.
func leadLayout(leads [][]byte) (indent string, multiline, found bool) {
	if len(leads) == 0 {
		return "", false, false
	}
	for _, l := range leads {
		if ind, ok := lineIndent(l); ok {
			return ind, true, true
		}
	}
	return "", false, true
}

func firstLead(leads [][]byte) []byte {
	if len(leads) == 0 {
		return nil
	}
	return leads[0]
}

// lineIndent returns the run of spaces and tabs after the last newline in extra.
func lineIndent(extra []byte) (string, bool) {
	i := bytes.LastIndexByte(extra, '\n')
	if i < 0 {
		return "", false
	}
	rest := extra[i+1:]
	n := 0
	for n < len(rest) && (rest[n] == ' ' || rest[n] == '\t') {
		n++
	}
	return string(rest[:n]), true
}

func lineIndentOr(extra []byte, fallback string) string {
	if ind, ok := lineIndent(extra); ok {
		return ind
	}
	return fallback
}

func hasComment(extra []byte) bool {
	return bytes.Contains(extra, []byte("//")) || bytes.Contains(extra, []byte("/*"))
}

// mergeExtra joins a removed child's leading extra (which holds comments) with the extra that
// follows it, so the comments keep their own lines.
func mergeExtra(removed, next []byte) []byte {
	head := bytes.TrimRight(removed, " \t")
	return append(bytes.Clone(head), bytes.TrimLeft(next, "\r\n")...)
}

// standardJSON returns v as canonical JSON: comments and trailing commas stripped, object keys sorted.
func standardJSON(v hujson.Value) ([]byte, error) {
	c := v.Clone()
	c.Standardize()
	var x any
	if err := json.Unmarshal(c.Pack(), &x); err != nil {
		return nil, errs.WrapMessage("failed to decode JSON value", err)
	}
	return marshalNoEscape(x)
}

// canonicalJSON returns val marshaled the way standardJSON would return it from a file.
func canonicalJSON(val any) ([]byte, error) {
	b, err := marshalNoEscape(val)
	if err != nil {
		return nil, err
	}
	var x any
	if err := json.Unmarshal(b, &x); err != nil {
		return nil, errs.WrapMessage("failed to decode JSON value", err)
	}
	return marshalNoEscape(x)
}

func marshalNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, errs.WrapMessage("failed to encode JSON value", err)
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// shortHash is the 16-hex-digit SHA-256 prefix stored as an entry hash.
func shortHash(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])[:16]
}
