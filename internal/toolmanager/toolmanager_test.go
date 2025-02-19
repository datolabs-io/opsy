package toolmanager

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"testing"

	"github.com/datolabs-io/sredo/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNew tests the creation of a new tool manager with various options.
func TestNew(t *testing.T) {
	t.Run("creates default tool manager", func(t *testing.T) {
		tm := New()
		assert.NotNil(t, tm)
		assert.Equal(t, "tools", tm.dir)
		assert.NotNil(t, tm.tools)
	})

	t.Run("creates tool manager with options", func(t *testing.T) {
		logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
		cfg := config.New().GetConfig()
		ctx := context.Background()

		tm := New(
			WithLogger(logger),
			WithConfig(cfg),
			WithContext(ctx),
			WithDirectory("testdata"),
		)

		assert.NotNil(t, tm)
		assert.Equal(t, ".", tm.dir)
		assert.Equal(t, cfg, tm.cfg)
		assert.Equal(t, ctx, tm.ctx)
	})
}

// TestLoadTools tests loading tools from the filesystem.
func TestLoadTools(t *testing.T) {
	t.Run("loads valid tools", func(t *testing.T) {
		tm := New(WithDirectory("testdata"))
		err := tm.LoadTools()
		require.NoError(t, err)

		tools := tm.GetTools()
		assert.Len(t, tools, 3) // Should load test_tool.yaml, executable_tool.yaml and exec tool

		tool, ok := tools["test_tool"]
		require.True(t, ok)
		assert.Equal(t, "Test Tool", tool.GetDisplayName())
		assert.Equal(t, "A tool for testing purposes", tool.GetDescription())

		// Verify executable tool is loaded
		executableTool, ok := tools["executable_tool"]
		require.True(t, ok)
		assert.Equal(t, "Executable Tool", executableTool.GetDisplayName())
		assert.Equal(t, "A test tool with executable", executableTool.GetDescription())

		// Verify exec tool is loaded
		execTool, ok := tools[ExecToolName]
		require.True(t, ok)
		assert.Equal(t, "Exec", execTool.GetDisplayName())
	})

	t.Run("handles invalid tool definitions", func(t *testing.T) {
		// The invalid tool should be skipped during loading
		tm := New(WithDirectory("testdata"))
		err := tm.LoadTools()
		require.NoError(t, err)

		_, err = tm.GetTool("invalid_tool")
		assert.ErrorContains(t, err, ErrToolNotFound)
	})

	t.Run("validates required fields", func(t *testing.T) {
		def := &toolDefinition{
			DisplayName: "Test Tool",
			Description: "Test Description",
			Inputs: map[string]toolInput{
				"test_input": {
					Description: "Test Input", // Missing type
				},
			},
		}
		err := validateToolDefinition(def)
		assert.ErrorContains(t, err, fmt.Sprintf("%s: %q", ErrToolInputMissingType, "test_input"))

		def.Inputs["test_input"] = toolInput{
			Type: "string", // Missing description
		}
		err = validateToolDefinition(def)
		assert.ErrorContains(t, err, fmt.Sprintf("%s: %q", ErrToolInputMissingDescription, "test_input"))

		def.DisplayName = ""
		err = validateToolDefinition(def)
		assert.ErrorContains(t, err, ErrToolMissingDisplayName)

		def.DisplayName = "Test Tool"
		def.Description = ""
		err = validateToolDefinition(def)
		assert.ErrorContains(t, err, ErrToolMissingDescription)
	})

	t.Run("handles non-existent directory", func(t *testing.T) {
		tm := New(WithDirectory("nonexistent"))
		err := tm.LoadTools()
		assert.ErrorContains(t, err, ErrLoadingTools)
	})

	t.Run("handles non-yaml files", func(t *testing.T) {
		// Create a temporary directory for test files
		tmpDir := t.TempDir()

		// Create a non-YAML file
		err := os.WriteFile(filepath.Join(tmpDir, "not_a_tool.txt"), []byte("not a tool"), 0644)
		require.NoError(t, err)

		// Create a valid YAML file
		err = os.WriteFile(filepath.Join(tmpDir, "valid_tool.yaml"), []byte(`
display_name: "Valid Tool"
description: "A valid tool"
inputs:
  test_input:
    type: "string"
    description: "Test input"
`), 0644)
		require.NoError(t, err)

		tm := New(WithDirectory(tmpDir))
		err = tm.LoadTools()
		require.NoError(t, err)

		tools := tm.GetTools()
		assert.Len(t, tools, 2) // Should only load valid_tool.yaml and exec tool
	})

	t.Run("handles empty directory", func(t *testing.T) {
		tmpDir := t.TempDir()
		tm := New(WithDirectory(tmpDir))
		err := tm.LoadTools()
		require.NoError(t, err)
		assert.Len(t, tm.GetTools(), 1) // Should only have exec tool
	})

	t.Run("handles directory with only invalid tools", func(t *testing.T) {
		tmpDir := t.TempDir()

		// Create an invalid YAML file
		err := os.WriteFile(filepath.Join(tmpDir, "invalid.yaml"), []byte("invalid: yaml: content"), 0644)
		require.NoError(t, err)

		tm := New(WithDirectory(tmpDir))
		err = tm.LoadTools()
		require.NoError(t, err)
		assert.Len(t, tm.GetTools(), 1) // Should only have exec tool
	})
}

// TestGetTool tests retrieving specific tools.
func TestGetTool(t *testing.T) {
	tm := New(WithDirectory("testdata"))
	require.NoError(t, tm.LoadTools())

	t.Run("gets existing tool", func(t *testing.T) {
		tool, err := tm.GetTool("test_tool")
		require.NoError(t, err)
		assert.Equal(t, "Test Tool", tool.GetDisplayName())
	})

	t.Run("gets exec tool", func(t *testing.T) {
		tool, err := tm.GetTool(ExecToolName)
		require.NoError(t, err)
		assert.Equal(t, "Exec", tool.GetDisplayName())
	})

	t.Run("returns error for non-existent tool", func(t *testing.T) {
		_, err := tm.GetTool("nonexistent")
		assert.ErrorContains(t, err, ErrToolNotFound)
	})
}

// TestGetTools tests retrieving all tools.
func TestGetTools(t *testing.T) {
	tm := New(WithDirectory("testdata"))
	require.NoError(t, tm.LoadTools())

	tools := tm.GetTools()
	assert.Len(t, tools, 3) // Should have test_tool, executable_tool and exec tool

	// Verify test_tool
	testTool, ok := tools["test_tool"]
	require.True(t, ok, "test_tool should be present")
	assert.Equal(t, "Test Tool", testTool.GetDisplayName())

	// Verify executable_tool
	executableTool, ok := tools["executable_tool"]
	require.True(t, ok, "executable_tool should be present")
	assert.Equal(t, "Executable Tool", executableTool.GetDisplayName())

	// Verify exec tool
	execTool, ok := tools[ExecToolName]
	require.True(t, ok, "exec tool should be present")
	assert.Equal(t, "Exec", execTool.GetDisplayName())
}
