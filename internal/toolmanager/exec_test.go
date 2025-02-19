package toolmanager

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/datolabs-io/sredo/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTestLogger creates a logger suitable for testing
func newTestLogger() *slog.Logger {
	return slog.New(slog.DiscardHandler)
}

// newTestConfig creates a configuration suitable for testing
func newTestConfig() *config.ToolsConfiguration {
	return &config.ToolsConfiguration{
		Timeout: 30,
		Exec: config.ExecToolConfiguration{
			Timeout: 10,
			Shell:   "/bin/bash",
		},
	}
}

// TestNewExecTool tests the creation of a new exec tool.
func TestNewExecTool(t *testing.T) {
	logger := newTestLogger()
	cfg := newTestConfig()
	tool := newExecTool(logger, cfg)

	t.Run("initializes with correct values", func(t *testing.T) {
		assert.Equal(t, ExecToolName, tool.GetName())
		assert.Equal(t, "Exec", tool.GetDisplayName())
		assert.Equal(t, "Executes the provided shell command.", tool.GetDescription())
		assert.NotNil(t, tool.GetInputSchema())
		assert.Equal(t, cfg, tool.config)
	})

	t.Run("has correct input schema", func(t *testing.T) {
		schema := tool.GetInputSchema()
		require.NotNil(t, schema)

		// Verify command input
		commandProp, ok := schema.Properties.Get(InputCommand)
		require.True(t, ok)
		assert.Equal(t, "string", commandProp.Type)
		assert.Equal(t, "The shell command, including all the arguments, to execute", commandProp.Description)
		assert.NotEmpty(t, commandProp.Examples)
	})
}

// TestExecTool_Execute tests the execution of commands through the exec tool.
func TestExecTool_Execute(t *testing.T) {
	logger := newTestLogger()
	cfg := newTestConfig()
	tool := newExecTool(logger, cfg)

	t.Run("executes simple command successfully", func(t *testing.T) {
		inputs := map[string]any{
			InputCommand:          "echo -n 'test'",
			InputWorkingDirectory: ".",
		}
		output, err := tool.Execute(inputs, context.Background())
		require.NoError(t, err)
		assert.NotNil(t, output)
		assert.Equal(t, ExecToolName, output.Tool)
		assert.Equal(t, "test", output.Result)
		assert.False(t, output.IsError)
		assert.NotNil(t, output.ExecutedCommand)
		assert.Equal(t, "echo -n 'test'", output.ExecutedCommand.Command)
		assert.Equal(t, ".", output.ExecutedCommand.WorkingDirectory)
		assert.Equal(t, 0, output.ExecutedCommand.ExitCode)
		assert.False(t, output.ExecutedCommand.StartedAt.IsZero())
		assert.False(t, output.ExecutedCommand.CompletedAt.IsZero())
		assert.True(t, output.ExecutedCommand.StartedAt.Before(output.ExecutedCommand.CompletedAt))
	})

	t.Run("handles command error", func(t *testing.T) {
		inputs := map[string]any{
			InputCommand:          "nonexistentcommand",
			InputWorkingDirectory: ".",
		}
		output, err := tool.Execute(inputs, context.Background())
		assert.Error(t, err)
		assert.Equal(t, "exit status 127", err.Error())
		assert.NotNil(t, output)
		assert.Equal(t, ExecToolName, output.Tool)
		assert.Contains(t, output.Result, "command not found")
		assert.True(t, output.IsError)
		assert.NotNil(t, output.ExecutedCommand)
		assert.Equal(t, "nonexistentcommand", output.ExecutedCommand.Command)
		assert.Equal(t, ".", output.ExecutedCommand.WorkingDirectory)
		assert.Equal(t, 127, output.ExecutedCommand.ExitCode)
	})

	t.Run("validates command input type", func(t *testing.T) {
		inputs := map[string]any{
			InputCommand:          123, // Invalid type
			InputWorkingDirectory: ".",
		}
		output, err := tool.Execute(inputs, context.Background())
		assert.Error(t, err)
		assert.Contains(t, err.Error(), ErrInvalidToolInputType)
		assert.Nil(t, output)
	})

	t.Run("validates working directory input type", func(t *testing.T) {
		inputs := map[string]any{
			InputCommand:          "echo 'test'",
			InputWorkingDirectory: 123, // Invalid type
		}
		output, err := tool.Execute(inputs, context.Background())
		assert.Error(t, err)
		assert.Contains(t, err.Error(), ErrInvalidToolInputType)
		assert.Nil(t, output)
	})
}

// TestExecTool_Timeout tests the timeout functionality of the exec tool.
func TestExecTool_Timeout(t *testing.T) {
	logger := newTestLogger()

	t.Run("uses exec timeout when set", func(t *testing.T) {
		cfg := &config.ToolsConfiguration{
			Timeout: 30,
			Exec: config.ExecToolConfiguration{
				Timeout: 2,
				Shell:   "/bin/bash",
			},
		}
		tool := newExecTool(logger, cfg)
		timeout := tool.getTimeout()
		assert.Equal(t, 2*time.Second, timeout)
	})

	t.Run("uses global timeout when exec timeout is zero", func(t *testing.T) {
		cfg := &config.ToolsConfiguration{
			Timeout: 30,
			Exec: config.ExecToolConfiguration{
				Timeout: 0,
				Shell:   "/bin/bash",
			},
		}
		tool := newExecTool(logger, cfg)
		timeout := tool.getTimeout()
		assert.Equal(t, 30*time.Second, timeout)
	})

	t.Run("command times out", func(t *testing.T) {
		cfg := &config.ToolsConfiguration{
			Timeout: 30,
			Exec: config.ExecToolConfiguration{
				Timeout: 1,
				Shell:   "/bin/bash",
			},
		}
		tool := newExecTool(logger, cfg)
		inputs := map[string]any{
			InputCommand:          "sleep 5",
			InputWorkingDirectory: ".",
		}
		output, err := tool.Execute(inputs, context.Background())
		assert.Error(t, err)
		assert.Equal(t, "signal: killed", err.Error())
		assert.NotNil(t, output)
		assert.True(t, output.IsError)
		assert.Empty(t, output.Result)
		assert.NotNil(t, output.ExecutedCommand)
		assert.Equal(t, "sleep 5", output.ExecutedCommand.Command)
		assert.Equal(t, ".", output.ExecutedCommand.WorkingDirectory)
	})
}

// TestExecTool_WorkingDirectory tests working directory functionality.
func TestExecTool_WorkingDirectory(t *testing.T) {
	logger := newTestLogger()
	cfg := newTestConfig()
	tool := newExecTool(logger, cfg)

	t.Run("executes in specified working directory", func(t *testing.T) {
		inputs := map[string]any{
			InputCommand:          "pwd",
			InputWorkingDirectory: "/tmp",
		}
		output, err := tool.Execute(inputs, context.Background())
		require.NoError(t, err)
		assert.NotNil(t, output)
		assert.Equal(t, ExecToolName, output.Tool)
		assert.Contains(t, output.Result.(string), "/tmp")
		assert.False(t, output.IsError)
		assert.NotNil(t, output.ExecutedCommand)
		assert.Equal(t, "/tmp", output.ExecutedCommand.WorkingDirectory)
		assert.Equal(t, 0, output.ExecutedCommand.ExitCode)
	})

	t.Run("handles nonexistent working directory", func(t *testing.T) {
		inputs := map[string]any{
			InputCommand:          "pwd",
			InputWorkingDirectory: "/nonexistent/directory",
		}
		output, err := tool.Execute(inputs, context.Background())
		assert.Error(t, err)
		assert.Equal(t, "fork/exec /bin/bash: no such file or directory", err.Error())
		assert.NotNil(t, output)
		assert.True(t, output.IsError)
		assert.Empty(t, output.Result)
		assert.NotNil(t, output.ExecutedCommand)
		assert.Equal(t, "/nonexistent/directory", output.ExecutedCommand.WorkingDirectory)
		assert.NotEqual(t, 0, output.ExecutedCommand.ExitCode)
	})
}

// TestExecTool_Timestamps tests the timestamp functionality of executed commands.
func TestExecTool_Timestamps(t *testing.T) {
	logger := newTestLogger()
	cfg := newTestConfig()
	tool := newExecTool(logger, cfg)

	t.Run("timestamps are set for quick commands", func(t *testing.T) {
		inputs := map[string]any{
			InputCommand:          "echo 'quick'",
			InputWorkingDirectory: ".",
		}
		output, err := tool.Execute(inputs, context.Background())
		require.NoError(t, err)
		assert.NotNil(t, output.ExecutedCommand)

		// Verify timestamps are set
		assert.False(t, output.ExecutedCommand.StartedAt.IsZero())
		assert.False(t, output.ExecutedCommand.CompletedAt.IsZero())

		// Verify timestamp order
		assert.True(t, output.ExecutedCommand.StartedAt.Before(output.ExecutedCommand.CompletedAt))

		// Verify reasonable duration (should be very quick)
		duration := output.ExecutedCommand.CompletedAt.Sub(output.ExecutedCommand.StartedAt)
		assert.Less(t, duration.Milliseconds(), int64(1000), "Quick command should complete in less than 1 second")
	})

	t.Run("timestamps are set for longer running commands", func(t *testing.T) {
		inputs := map[string]any{
			InputCommand:          "sleep 0.5",
			InputWorkingDirectory: ".",
		}
		output, err := tool.Execute(inputs, context.Background())
		require.NoError(t, err)
		assert.NotNil(t, output.ExecutedCommand)

		// Verify timestamps are set
		assert.False(t, output.ExecutedCommand.StartedAt.IsZero())
		assert.False(t, output.ExecutedCommand.CompletedAt.IsZero())

		// Verify timestamp order
		assert.True(t, output.ExecutedCommand.StartedAt.Before(output.ExecutedCommand.CompletedAt))

		// Verify duration is at least the sleep time
		duration := output.ExecutedCommand.CompletedAt.Sub(output.ExecutedCommand.StartedAt)
		assert.GreaterOrEqual(t, duration.Milliseconds(), int64(500), "Command should take at least 500ms")
	})

	t.Run("timestamps are set even when command fails", func(t *testing.T) {
		inputs := map[string]any{
			InputCommand:          "nonexistentcommand",
			InputWorkingDirectory: ".",
		}
		output, err := tool.Execute(inputs, context.Background())
		assert.Error(t, err)
		assert.NotNil(t, output.ExecutedCommand)

		// Verify timestamps are set even for failed commands
		assert.False(t, output.ExecutedCommand.StartedAt.IsZero())
		assert.False(t, output.ExecutedCommand.CompletedAt.IsZero())
		assert.True(t, output.ExecutedCommand.StartedAt.Before(output.ExecutedCommand.CompletedAt))
	})
}
