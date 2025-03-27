package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/datolabs-io/opsy/assets"
	"github.com/datolabs-io/opsy/internal/config"
	"github.com/datolabs-io/opsy/internal/tool"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
)

const (
	// ErrNoRunOptions is the error returned when no run options are provided.
	ErrNoRunOptions = "no run options provided"
	// ErrNoTaskProvided is the error returned when no task is provided.
	ErrNoTaskProvided = "no task provided"

	// StatusReady is the status of the agent when it is ready to run.
	StatusReady = "Ready"
	// StatusRunning is the status of the agent when it is running.
	StatusRunning = "Running"
	// StatusFinished is the status of the agent when it has finished.
	StatusFinished = "Finished"
	// StatusError is the status of the agent when it has encountered an error.
	StatusError = "Error"
)

// Status is the status of the agent.
type Status string

// Agent is a struct that contains the state of the agent.
type Agent struct {
	client        anthropic.Client
	ctx           context.Context
	cfg           config.Configuration
	logger        *slog.Logger
	communication *Communication
}

// Message is a struct that contains a message from the agent.
type Message struct {
	// Tool is the name of the tool that sent the message.
	Tool string
	// Message is the message from the tool.
	Message string
	// Timestamp is the timestamp when the message was sent.
	Timestamp time.Time
}

// Communication is a struct that contains the communication channels for the agent.
type Communication struct {
	Commands chan tool.Command
	Messages chan Message
	Status   chan Status
}

// Option is a function that configures the Agent.
type Option func(*Agent)

const (
	// Name is the name of the agent.
	Name = "Opsy"
)

// New creates a new Agent.
func New(opts ...Option) *Agent {
	a := &Agent{
		ctx:    context.Background(),
		cfg:    config.New().GetConfig(),
		logger: slog.New(slog.DiscardHandler),
		communication: &Communication{
			Commands: make(chan tool.Command),
			Messages: make(chan Message),
			Status:   make(chan Status),
		},
	}

	for _, opt := range opts {
		opt(a)
	}

	if a.cfg.Anthropic.APIKey != "" {
		a.client = anthropic.NewClient(option.WithAPIKey(a.cfg.Anthropic.APIKey))
	}

	a.logger.WithGroup("config").With("max_tokens", a.cfg.Anthropic.MaxTokens).With("model", a.cfg.Anthropic.Model).
		With("temperature", a.cfg.Anthropic.Temperature).Debug("Agent initialized.")

	return a
}

// WithContext sets the context for the agent.
func WithContext(ctx context.Context) Option {
	return func(a *Agent) {
		a.ctx = ctx
	}
}

// WithConfig sets the configuration for the agent.
func WithConfig(cfg config.Configuration) Option {
	return func(a *Agent) {
		a.cfg = cfg
	}
}

// WithLogger sets the logger for the agent.
func WithLogger(logger *slog.Logger) Option {
	return func(a *Agent) {
		a.logger = logger.With("component", "agent")
	}
}

// WithClient sets the client for the agent.
func WithClient(client anthropic.Client) Option {
	return func(a *Agent) {
		a.client = client
	}
}

// WithCommunication sets the communication channels for the agent.
func WithCommunication(communication *Communication) Option {
	return func(a *Agent) {
		a.communication = communication
	}
}

// Run runs the agent with the given task and tools.
func (a *Agent) Run(opts *tool.RunOptions, ctx context.Context) ([]tool.Output, error) {
	if opts == nil {
		return nil, errors.New(ErrNoRunOptions)
	}

	if opts.Task == "" {
		return nil, errors.New(ErrNoTaskProvided)
	}

	if ctx == nil {
		ctx = a.ctx
	}

	prompt, err := assets.RenderAgentSystemPrompt(&assets.AgentSystemPromptData{
		Shell: a.cfg.Tools.Exec.Shell,
	})
	if err != nil {
		return nil, fmt.Errorf("%s: %w", assets.ErrToolRenderingPrompt, err)
	}

	if opts.Prompt != "" {
		prompt = opts.Prompt
	}

	logger := a.logger.With("task", opts.Task).With("tool", opts.Caller).With("tools.count", len(opts.Tools))
	logger.Debug("Agent running.")
	a.communication.Status <- StatusRunning

	output := []tool.Output{}
	messages := []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(opts.Task))}

	for {
		msg := anthropic.MessageNewParams{
			Model:       a.cfg.Anthropic.Model,
			MaxTokens:   a.cfg.Anthropic.MaxTokens,
			System:      []anthropic.TextBlockParam{{Text: prompt}},
			Messages:    messages,
			Tools:       convertTools(opts.Tools),
			Temperature: param.NewOpt(a.cfg.Anthropic.Temperature),
		}

		if len(opts.Tools) > 0 {
			msg.ToolChoice = anthropic.ToolChoiceUnionParam{
				OfToolChoiceAuto: &anthropic.ToolChoiceAutoParam{
					DisableParallelToolUse: param.NewOpt(true),
					Type:                   msg.ToolChoice.OfToolChoiceAuto.Type,
				},
			}
		}

		message, err := a.client.Messages.New(ctx, msg)

		if err != nil {
			// TODO(t-dabasinskas): Implement retry logic
			logger.With("error", err).Error("Failed to send message to Anthropic API.")
			return nil, err
		}

		toolResults := []anthropic.ContentBlockParamUnion{}
		for _, block := range message.Content {
			switch block := block.Type.(type) {
			case anthropic.TextBlock:
				a.communication.Messages <- Message{
					Tool:      opts.Caller,
					Message:   block.Text,
					Timestamp: time.Now(),
				}
				// TODO(t-dabasinskas): Remove this once we update UI
				logger.With("message", block.Text).Debug("Agent message.")
			case anthropic.ToolUseBlock:
				isError := false
				resultBlockContent := ""
				toolInputs := map[string]any{}

				if err := json.Unmarshal(block.Input, &toolInputs); err != nil {
					logger.With("error", err).Error("Failed to unmarshal tool inputs.")
					continue
				}

				var toolOutput *tool.Output
				tool, ok := opts.Tools[block.Name]
				if !ok {
					logger.With("tool_name", block.Name).Warn("Tool not found, skipping.")
					continue
				}

				toolOutput, err = tool.Execute(toolInputs, ctx)
				if err != nil {
					logger.With("error", err).Error("Failed to execute tool.")
					isError = true
				}

				if toolOutput == nil {
					logger.With("tool_name", block.Name).Warn("Tool has no output, skipping.")
					continue
				}

				output = append(output, *toolOutput)

				// Handle messages from all the tools except the Exec:
				if toolOutput.Result != "" && toolOutput.ExecutedCommand == nil {
					resultBlockContent = toolOutput.Result
					a.communication.Messages <- Message{
						Tool:      opts.Caller,
						Message:   toolOutput.Result,
						Timestamp: time.Now(),
					}
				}
				logger.With("output", toolOutput).Warn(">>>>Tool result.")

				// Handle messages from the Exec tool:
				if toolOutput.ExecutedCommand != nil {
					resultBlockContent = toolOutput.ExecutedCommand.Output
					isError = toolOutput.ExecutedCommand.ExitCode != 0
					a.communication.Commands <- *toolOutput.ExecutedCommand
				}

				resultBlock := anthropic.NewToolResultBlock(block.ID, resultBlockContent, isError)
				toolResults = append(toolResults, resultBlock)
			}
		}

		messages = append(messages, message.ToParam())
		if len(toolResults) == 0 {
			break
		}

		messages = append(messages, anthropic.NewUserMessage(toolResults...))
	}

	return output, nil
}

// convertTools converts the tools to the format required by the Anthropic SDK.
func convertTools(tools map[string]tool.Tool) (anthropicTools []anthropic.ToolUnionParam) {
	for _, t := range tools {
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.GetName(),
				Description: param.NewOpt(t.GetDescription()),
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: t.GetInputSchema().Properties,
				},
			},
		})
	}
	return
}
