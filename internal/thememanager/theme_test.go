package thememanager

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gopkg.in/yaml.v3"
)

// TestTheme_UnmarshalYAML verifies theme YAML unmarshaling:
// - Valid theme with all required colors
// - Theme missing required colors
// - Invalid YAML syntax
// - Invalid color format
func TestTheme_UnmarshalYAML(t *testing.T) {
	tests := []struct {
		name    string
		yaml    string
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid theme",
			yaml: `
base:
  base00: "#1A1B26"
  base01: "#24283B"
  base02: "#292E42"
  base03: "#565F89"
  base04: "#A9B1D6"
accent:
  accent0: "#FF9E64"
  accent1: "#9ECE6A"
  accent2: "#7AA2F7"`,
			wantErr: false,
		},
		{
			name: "missing color",
			yaml: `
base:
  base00: "#1A1B26"
  base01: "#24283B"
accent:
  accent0: "#FF9E64"
  accent1: "#9ECE6A"
  accent2: "#7AA2F7"`,
			wantErr: true,
			errMsg:  ErrMissingColors,
		},
		{
			name: "invalid color format - missing #",
			yaml: `
base:
  base00: "1A1B26"
  base01: "#24283B"
  base02: "#292E42"
  base03: "#565F89"
  base04: "#A9B1D6"
accent:
  accent0: "#FF9E64"
  accent1: "#9ECE6A"
  accent2: "#7AA2F7"`,
			wantErr: true,
			errMsg:  ErrInvalidColor,
		},
		{
			name: "invalid color format - not hex",
			yaml: `
base:
  base00: "#ZZZZZZ"
  base01: "#24283B"
  base02: "#292E42"
  base03: "#565F89"
  base04: "#A9B1D6"
accent:
  accent0: "#FF9E64"
  accent1: "#9ECE6A"
  accent2: "#7AA2F7"`,
			wantErr: true,
			errMsg:  ErrInvalidColor,
		},
		{
			name:    "invalid yaml",
			yaml:    `{`, // Invalid YAML syntax
			wantErr: true,
			errMsg:  ErrDecodingTheme,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var node yaml.Node
			err := yaml.Unmarshal([]byte(tt.yaml), &node)
			if err != nil {
				if tt.wantErr && tt.errMsg == ErrDecodingTheme {
					assert.Error(t, err)
					return
				}
				t.Fatalf("failed to unmarshal YAML: %v", err)
			}

			var theme Theme
			err = theme.UnmarshalYAML(&node)

			if tt.wantErr {
				assert.Error(t, err)
				if err != nil {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
				return
			}
			assert.NoError(t, err)
		})
	}
}

// TestTheme_ColorValidation verifies that all theme colors are properly validated:
// - All colors must be valid hex color codes
// - Colors must start with '#'
func TestTheme_ColorValidation(t *testing.T) {
	validTheme := `
base:
  base00: "#1A1B26"
  base01: "#24283B"
  base02: "#292E42"
  base03: "#565F89"
  base04: "#A9B1D6"
accent:
  accent0: "#FF9E64"
  accent1: "#9ECE6A"
  accent2: "#7AA2F7"`

	var theme Theme
	var node yaml.Node
	err := yaml.Unmarshal([]byte(validTheme), &node)
	assert.NoError(t, err)

	err = theme.UnmarshalYAML(&node)
	assert.NoError(t, err)

	// Test base colors
	assert.Equal(t, "#1A1B26", string(theme.BaseColors.Base00))
	assert.Equal(t, "#24283B", string(theme.BaseColors.Base01))
	assert.Equal(t, "#292E42", string(theme.BaseColors.Base02))
	assert.Equal(t, "#565F89", string(theme.BaseColors.Base03))
	assert.Equal(t, "#A9B1D6", string(theme.BaseColors.Base04))

	// Test accent colors
	assert.Equal(t, "#FF9E64", string(theme.AccentColors.Accent0))
	assert.Equal(t, "#9ECE6A", string(theme.AccentColors.Accent1))
	assert.Equal(t, "#7AA2F7", string(theme.AccentColors.Accent2))
}

// TestTheme_ColorFormat verifies color format validation:
// - Colors must be valid hex codes
// - Colors must start with '#'
func TestTheme_ColorFormat(t *testing.T) {
	tests := []struct {
		name    string
		color   string
		wantErr bool
		errMsg  string
	}{
		{
			name:    "valid hex color",
			color:   "#1A1B26",
			wantErr: false,
		},
		{
			name:    "missing hash",
			color:   "1A1B26",
			wantErr: true,
			errMsg:  ErrInvalidColor,
		},
		{
			name:    "invalid hex",
			color:   "#ZZZZZZ",
			wantErr: true,
			errMsg:  ErrInvalidColor,
		},
		{
			name:    "too short",
			color:   "#1A1",
			wantErr: true,
			errMsg:  ErrInvalidColor,
		},
		{
			name:    "too long",
			color:   "#1A1B26FF",
			wantErr: true,
			errMsg:  ErrInvalidColor,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yamlStr := fmt.Sprintf(`
base:
  base00: "%s"
  base01: "#24283B"
  base02: "#292E42"
  base03: "#565F89"
  base04: "#A9B1D6"
accent:
  accent0: "#FF9E64"
  accent1: "#9ECE6A"
  accent2: "#7AA2F7"`, tt.color)

			var node yaml.Node
			err := yaml.Unmarshal([]byte(yamlStr), &node)
			assert.NoError(t, err)

			var theme Theme
			err = theme.UnmarshalYAML(&node)

			if tt.wantErr {
				assert.Error(t, err)
				if err != nil {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
