package skills

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSkillFrontmatterIsPlainYAML guards the generated frontmatter: a ": " inside a plain YAML
// scalar is a parse error, so a host would fail to load the installed skill.
func TestSkillFrontmatterIsPlainYAML(t *testing.T) {
	t.Parallel()
	all, err := All()
	require.NoError(t, err)
	require.Len(t, all, len(catalog))
	for _, sk := range all {
		t.Run(sk.Name, func(t *testing.T) {
			t.Parallel()
			assert.NotContains(t, sk.Description, ": ")
			assert.False(t, strings.ContainsAny(sk.Description[:1], "[]{}>|*&!%#`@,'\"-?:"), "description starts with a YAML indicator")
			assert.True(t, strings.HasPrefix(sk.Content(), "---\nname: "+sk.DirName()+"\ndescription: "))
		})
	}
}
