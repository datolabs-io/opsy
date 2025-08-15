package agent

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/datolabs-io/opsy/internal/config"
	"github.com/datolabs-io/opsy/internal/tool"
	"github.com/invopop/jsonschema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	orderedmap "github.com/wk8/go-ordered-map/v2"
)

// mockTool implements the tool.Tool interface for testing
type mockTool struct {
	name        string
	displayName string
	description string
	schema      *jsonschema.Schema
	output      *tool.Output
	err         error
}

func (t *mockTool) GetName() string                    { return t.name }
func (t *mockTool) GetDisplayName() string             { return t.displayName }
func (t *mockTool) GetDescription() string             { return t.description }
func (t *mockTool) GetInputSchema() *jsonschema.Schema { return t.schema }
func (t *mockTool) Execute(inputs map[string]any, ctx context.Context) (*tool.Output, error) {
	return t.output, t.err
}

// TestNew tests agent creation and options
func TestNew(t *testing.T) {
	t.Run("creates default agent", func(t *testing.T) {
		agent := New()
		assert.NotNil(t, agent)
		assert.NotNil(t, agent.ctx)
		assert.NotNil(t, agent.cfg)
		assert.NotNil(t, agent.logger)
		assert.NotNil(t, agent.communication)
		assert.Nil(t, agent.client) // No API key set
	})

	t.Run("applies options", func(t *testing.T) {
		ctx := context.Background()
		cfg := config.New().GetConfig()
		logger := slog.New(slog.NewTextHandler(nil, nil))
		comm := &Communication{
			Commands: make(chan tool.Command),
			Messages: make(chan Message),
			Status:   make(chan Status),
		}

		agent := New(
			WithContext(ctx),
			WithConfig(cfg),
			WithLogger(logger),
			WithCommunication(comm),
		)

		assert.Equal(t, ctx, agent.ctx)
		assert.Equal(t, cfg, agent.cfg)
		assert.Equal(t, comm, agent.communication)
		assert.Nil(t, agent.client) // Agent without API key should have nil client
	})

	t.Run("creates client when API key provided", func(t *testing.T) {
		cfg := config.New().GetConfig()
		cfg.Anthropic.APIKey = "test-key"
		agent := New(WithConfig(cfg))
		assert.NotNil(t, agent.client)
		// Verify client is properly initialized by checking its type
		assert.IsType(t, &anthropic.Client{}, agent.client)
	})
}

// TestConvertTools tests tool conversion for Anthropic API
func TestConvertTools(t *testing.T) {
	t.Run("converts single tool", func(t *testing.T) {
		properties := orderedmap.New[string, *jsonschema.Schema]()
		properties.Set("test", &jsonschema.Schema{Type: "string"})

		schema := &jsonschema.Schema{
			Type:       "object",
			Properties: properties,
		}

		tools := map[string]tool.Tool{
			"test": &mockTool{
				name:        "test",
				displayName: "Test Tool",
				description: "A test tool",
				schema:      schema,
			},
		}

		anthropicTools := convertTools(tools)
		require.Len(t, anthropicTools, 1)

		toolParam := anthropicTools[0].OfTool
		require.NotNil(t, toolParam)
		assert.Equal(t, "test", toolParam.Name)
		assert.Equal(t, "A test tool", toolParam.Description.Value)
		assert.NotNil(t, toolParam.InputSchema)
	})

	t.Run("converts multiple tools", func(t *testing.T) {
		properties := orderedmap.New[string, *jsonschema.Schema]()
		properties.Set("param", &jsonschema.Schema{Type: "string"})

		schema := &jsonschema.Schema{
			Type:       "object",
			Properties: properties,
		}

		tools := map[string]tool.Tool{
			"tool1": &mockTool{
				name:        "tool1",
				displayName: "Tool One",
				description: "First test tool",
				schema:      schema,
			},
			"tool2": &mockTool{
				name:        "tool2",
				displayName: "Tool Two",
				description: "Second test tool",
				schema:      schema,
			},
		}

		anthropicTools := convertTools(tools)
		require.Len(t, anthropicTools, 2)

		// Verify both tools are present with correct values
		foundTool1 := false
		foundTool2 := false

		for _, toolUnion := range anthropicTools {
			toolParam := toolUnion.OfTool
			require.NotNil(t, toolParam)

			name := toolParam.Name
			switch name {
			case "tool1":
				foundTool1 = true
				assert.Equal(t, "First test tool", toolParam.Description.Value)
				assert.NotNil(t, toolParam.InputSchema)
			case "tool2":
				foundTool2 = true
				assert.Equal(t, "Second test tool", toolParam.Description.Value)
				assert.NotNil(t, toolParam.InputSchema)
			}
		}

		assert.True(t, foundTool1, "tool1 should be present")
		assert.True(t, foundTool2, "tool2 should be present")
	})

	t.Run("handles empty tools map", func(t *testing.T) {
		tools := map[string]tool.Tool{}
		anthropicTools := convertTools(tools)
		assert.Empty(t, anthropicTools)
	})
}

// TestCommunication tests the communication channels
func TestCommunication(t *testing.T) {
	t.Run("sends and receives messages", func(t *testing.T) {
		comm := &Communication{
			Commands: make(chan tool.Command),
			Messages: make(chan Message),
			Status:   make(chan Status),
		}

		agent := New(WithCommunication(comm))
		assert.NotNil(t, agent.communication)

		// Test message channel
		go func() {
			comm.Messages <- Message{
				Tool:      "test",
				Message:   "test message",
				Timestamp: time.Now(),
			}
			close(comm.Messages)
		}()

		msg := <-comm.Messages
		assert.Equal(t, "test message", msg.Message)
		assert.Equal(t, "test", msg.Tool)

		// Test status channel
		go func() {
			comm.Status <- Status(StatusRunning)
			close(comm.Status)
		}()

		status := <-comm.Status
		assert.Equal(t, Status(StatusRunning), status)

		// Test command channel
		now := time.Now()
		cmd := tool.Command{
			Command:          "test command",
			WorkingDirectory: "/test/dir",
			ExitCode:         0,
			Output:           "test output",
			StartedAt:        now,
			CompletedAt:      now.Add(time.Second),
		}
		go func() {
			comm.Commands <- cmd
			close(comm.Commands)
		}()

		receivedCmd := <-comm.Commands
		assert.Equal(t, cmd, receivedCmd)
	})
}
