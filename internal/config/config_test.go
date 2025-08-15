package config

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupTestEnv(t *testing.T) (string, func()) {
	// Create a temporary directory for test config
	tempDir, err := os.MkdirTemp("", "opsy-test-*")
	require.NoError(t, err)

	// Save original environment variables
	origHome := os.Getenv("HOME")
	origAPIKey := os.Getenv("ANTHROPIC_API_KEY")
	origLogLevel := os.Getenv("OPSY_LOGGING_LEVEL")
	origExecShell := os.Getenv("OPSY_TOOLS_EXEC_SHELL")

	// Set up test environment
	if err := os.Setenv("HOME", tempDir); err != nil {
		t.Fatalf("failed to set HOME environment variable: %v", err)
	}
	if err := os.Unsetenv("ANTHROPIC_API_KEY"); err != nil { // Ensure API key is not set
		t.Fatalf("failed to unset ANTHROPIC_API_KEY environment variable: %v", err)
	}
	if err := os.Unsetenv("OPSY_LOGGING_LEVEL"); err != nil { // Ensure log level is not set
		t.Fatalf("failed to unset OPSY_LOGGING_LEVEL environment variable: %v", err)
	}
	if err := os.Unsetenv("OPSY_TOOLS_EXEC_SHELL"); err != nil { // Ensure shell is not set
		t.Fatalf("failed to unset OPSY_TOOLS_EXEC_SHELL environment variable: %v", err)
	}

	// Return cleanup function
	cleanup := func() {
		if err := os.RemoveAll(tempDir); err != nil {
			t.Logf("failed to remove temp directory: %v", err)
		}
		if err := os.Setenv("HOME", origHome); err != nil {
			t.Logf("failed to restore HOME environment variable: %v", err)
		}
		if err := os.Setenv("ANTHROPIC_API_KEY", origAPIKey); err != nil {
			t.Logf("failed to restore ANTHROPIC_API_KEY environment variable: %v", err)
		}
		if origLogLevel != "" {
			if err := os.Setenv("OPSY_LOGGING_LEVEL", origLogLevel); err != nil {
				t.Logf("failed to restore OPSY_LOGGING_LEVEL environment variable: %v", err)
			}
		} else {
			if err := os.Unsetenv("OPSY_LOGGING_LEVEL"); err != nil {
				t.Logf("failed to unset OPSY_LOGGING_LEVEL environment variable: %v", err)
			}
		}
		if origExecShell != "" {
			if err := os.Setenv("OPSY_TOOLS_EXEC_SHELL", origExecShell); err != nil {
				t.Logf("failed to restore OPSY_TOOLS_EXEC_SHELL environment variable: %v", err)
			}
		} else {
			if err := os.Unsetenv("OPSY_TOOLS_EXEC_SHELL"); err != nil {
				t.Logf("failed to unset OPSY_TOOLS_EXEC_SHELL environment variable: %v", err)
			}
		}
		viper.Reset()
	}

	return tempDir, cleanup
}

// TestNewConfigManager verifies configuration manager creation:
// - Creates manager with default values
// - Properly binds environment variables
func TestNewConfigManager(t *testing.T) {
	tempDir, cleanup := setupTestEnv(t)
	defer cleanup()

	t.Run("creates new config manager with defaults", func(t *testing.T) {
		manager := New()
		assert.NotNil(t, manager)
		assert.Equal(t, tempDir, manager.homePath)

		// Verify default values are set in viper
		assert.Equal(t, filepath.Join(tempDir, ".opsy", "log.log"), viper.GetString("logging.path"))
		assert.Equal(t, "info", viper.GetString("logging.level"))
		assert.Equal(t, "claude-3-7-sonnet-latest", viper.GetString("anthropic.model"))
		assert.Equal(t, 0.7, viper.GetFloat64("anthropic.temperature"))
		assert.Equal(t, int64(1024), viper.GetInt64("anthropic.max_tokens"))
		assert.Equal(t, int64(120), viper.GetInt64("tools.timeout"))
		assert.Equal(t, int64(0), viper.GetInt64("tools.exec.timeout"))
		assert.Equal(t, "/bin/sh", viper.GetString("tools.exec.shell"))
	})

	t.Run("binds environment variables", func(t *testing.T) {
		if err := os.Setenv("ANTHROPIC_API_KEY", "test-api-key"); err != nil {
			t.Fatalf("failed to set ANTHROPIC_API_KEY environment variable: %v", err)
		}
		if err := os.Setenv("OPSY_LOGGING_LEVEL", "debug"); err != nil {
			t.Fatalf("failed to set OPSY_LOGGING_LEVEL environment variable: %v", err)
		}

		manager := New()
		assert.NotNil(t, manager)

		err := manager.LoadConfig()
		require.NoError(t, err)

		config := manager.GetConfig()
		assert.Equal(t, "test-api-key", config.Anthropic.APIKey)
		assert.Equal(t, "debug", config.Logging.Level)
	})
}

// TestLoadConfig_DefaultValues verifies default configuration loading:
// - Loads default values when no config file exists
// - Properly sets default values for all configuration fields
func TestLoadConfig_DefaultValues(t *testing.T) {
	_, cleanup := setupTestEnv(t)
	defer cleanup()

	if err := os.Setenv("ANTHROPIC_API_KEY", "test-api-key"); err != nil {
		t.Fatalf("failed to set ANTHROPIC_API_KEY environment variable: %v", err)
	}
	manager := New()
	err := manager.LoadConfig()
	require.NoError(t, err)

	config := manager.GetConfig()
	assert.Equal(t, "info", config.Logging.Level)
	assert.Equal(t, "claude-3-7-sonnet-latest", config.Anthropic.Model)
	assert.Equal(t, 0.7, config.Anthropic.Temperature)
	assert.Equal(t, int64(1024), config.Anthropic.MaxTokens)
	assert.Equal(t, int64(120), config.Tools.Timeout)
	assert.Equal(t, int64(0), config.Tools.Exec.Timeout)
	assert.Equal(t, "/bin/sh", config.Tools.Exec.Shell)
}

// TestLoadConfig_CustomValues verifies custom configuration loading:
// - Loads custom values from config file
// - Properly overrides default values
// - Handles all configuration sections (UI, Logging, Anthropic)
func TestLoadConfig_CustomValues(t *testing.T) {
	tempDir, cleanup := setupTestEnv(t)
	defer cleanup()

	// Create custom config file
	configDir := filepath.Join(tempDir, ".opsy")
	require.NoError(t, os.MkdirAll(configDir, 0755))

	// Read test data
	testData, err := os.ReadFile("testdata/custom_config.yaml")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(configDir, "config.yaml"), testData, 0644))

	manager := New()
	err = manager.LoadConfig()
	require.NoError(t, err)

	config := manager.GetConfig()
	assert.Equal(t, "debug", config.Logging.Level)
	assert.Equal(t, "/custom/log/path", config.Logging.Path)
	assert.Equal(t, "claude-3-opus", config.Anthropic.Model)
	assert.Equal(t, 0.7, config.Anthropic.Temperature)
	assert.Equal(t, int64(2048), config.Anthropic.MaxTokens)
	assert.Equal(t, "custom_theme", config.UI.Theme)
	assert.Equal(t, int64(180), config.Tools.Timeout)
	assert.Equal(t, int64(90), config.Tools.Exec.Timeout)
	assert.Equal(t, "/bin/sh", config.Tools.Exec.Shell)
}

// TestLoadConfig_ValidationErrors verifies configuration validation:
// - Validates missing API key
// - Validates temperature range
// - Validates max tokens value
// - Validates log level values
func TestLoadConfig_ValidationErrors(t *testing.T) {
	tests := []struct {
		name        string
		configFile  string
		configData  []byte
		expectedErr string
	}{
		{
			name:        "missing API key",
			configData:  []byte("logging:\n  level: info"),
			expectedErr: "anthropic API key is required",
		},
		{
			name: "invalid temperature low",
			configData: []byte(`
anthropic:
  api_key: test-key
  temperature: -0.1
logging:
  level: info`),
			expectedErr: "anthropic temperature must be between 0 and 1",
		},
		{
			name: "invalid temperature high",
			configData: []byte(`
anthropic:
  api_key: test-key
  temperature: 1.1
logging:
  level: info`),
			expectedErr: "anthropic temperature must be between 0 and 1",
		},
		{
			name: "invalid max tokens",
			configData: []byte(`
anthropic:
  api_key: test-key
  max_tokens: 0
logging:
  level: info`),
			expectedErr: "anthropic max tokens must be greater than 0",
		},
		{
			name: "invalid log level",
			configData: []byte(`
anthropic:
  api_key: test-key
logging:
  level: invalid`),
			expectedErr: "invalid logging level",
		},
		{
			name: "missing shell",
			configData: []byte(`
anthropic:
  api_key: test-key
tools:
  exec:
    shell: ""`),
			expectedErr: "invalid exec shell",
		},
		{
			name: "invalid shell path",
			configData: []byte(`
anthropic:
  api_key: test-key
tools:
  exec:
    shell: "/nonexistent/shell"`),
			expectedErr: "invalid exec shell",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tempDir, cleanup := setupTestEnv(t)
			defer cleanup()

			// Create config file with test case content
			configDir := filepath.Join(tempDir, ".opsy")
			require.NoError(t, os.MkdirAll(configDir, 0755))
			require.NoError(t, os.WriteFile(filepath.Join(configDir, "config.yaml"), tt.configData, 0644))

			manager := New()
			err := manager.LoadConfig()
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
		})
	}
}

// TestLoadConfig_EnvironmentVariables verifies environment variable handling:
// - Environment variables override file configuration
// - Handles all supported environment variables
// - Properly converts environment variable types
func TestLoadConfig_EnvironmentVariables(t *testing.T) {
	_, cleanup := setupTestEnv(t)
	defer cleanup()

	// Set environment variables
	if err := os.Setenv("ANTHROPIC_API_KEY", "test-api-key"); err != nil {
		t.Fatalf("failed to set ANTHROPIC_API_KEY environment variable: %v", err)
	}
	if err := os.Setenv("OPSY_LOGGING_LEVEL", "debug"); err != nil {
		t.Fatalf("failed to set OPSY_LOGGING_LEVEL environment variable: %v", err)
	}
	if err := os.Setenv("OPSY_ANTHROPIC_MODEL", "claude-3-opus"); err != nil {
		t.Fatalf("failed to set OPSY_ANTHROPIC_MODEL environment variable: %v", err)
	}
	if err := os.Setenv("OPSY_ANTHROPIC_TEMPERATURE", "0.8"); err != nil {
		t.Fatalf("failed to set OPSY_ANTHROPIC_TEMPERATURE environment variable: %v", err)
	}
	defer func() {
		if err := os.Unsetenv("OPSY_LOGGING_LEVEL"); err != nil {
			t.Logf("failed to unset OPSY_LOGGING_LEVEL environment variable: %v", err)
		}
		if err := os.Unsetenv("OPSY_ANTHROPIC_MODEL"); err != nil {
			t.Logf("failed to unset OPSY_ANTHROPIC_MODEL environment variable: %v", err)
		}
		if err := os.Unsetenv("OPSY_ANTHROPIC_TEMPERATURE"); err != nil {
			t.Logf("failed to unset OPSY_ANTHROPIC_TEMPERATURE environment variable: %v", err)
		}
	}()

	manager := New()
	err := manager.LoadConfig()
	require.NoError(t, err)

	config := manager.GetConfig()
	assert.Equal(t, "debug", config.Logging.Level)
	assert.Equal(t, "claude-3-opus", config.Anthropic.Model)
	assert.Equal(t, 0.8, config.Anthropic.Temperature)
}

// TestGetLogger verifies logger creation:
// - Creates logger with valid configuration
// - Handles different log paths
// - Handles logger creation errors
func TestGetLogger(t *testing.T) {
	tempDir, cleanup := setupTestEnv(t)
	defer cleanup()

	tests := []struct {
		name        string
		logPath     string
		logLevel    string
		expectError bool
	}{
		{
			name:        "valid logger with debug level",
			logPath:     filepath.Join(tempDir, "test.log"),
			logLevel:    "debug",
			expectError: false,
		},
		{
			name:        "valid logger with info level",
			logPath:     filepath.Join(tempDir, "test.log"),
			logLevel:    "info",
			expectError: false,
		},
		{
			name:        "invalid log path",
			logPath:     "/nonexistent/directory/test.log",
			logLevel:    "info",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager := New()
			manager.configuration.Logging.Path = tt.logPath
			manager.configuration.Logging.Level = tt.logLevel

			logger, err := manager.GetLogger()
			if tt.expectError {
				assert.Error(t, err)
				assert.ErrorIs(t, err, ErrOpenLogFile)
				assert.Nil(t, logger)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, logger)

			// Clean up the log file
			if !tt.expectError {
				if err := os.Remove(tt.logPath); err != nil {
					t.Logf("failed to remove log file: %v", err)
				}
			}
		})
	}
}

// TestLoadConfig_DirectoryCreation verifies directory structure creation:
// - Creates required directories
// - Handles directory creation errors
// - Manages directory permissions
func TestLoadConfig_DirectoryCreation(t *testing.T) {
	t.Run("creates config and cache directories", func(t *testing.T) {
		tempDir, cleanup := setupTestEnv(t)
		defer cleanup()

		// Set required API key
		if err := os.Setenv("ANTHROPIC_API_KEY", "test-api-key"); err != nil {
			t.Fatalf("failed to set ANTHROPIC_API_KEY environment variable: %v", err)
		}

		manager := New()
		err := manager.LoadConfig()
		require.NoError(t, err)

		// Check if directories were created
		configDir := filepath.Join(tempDir, ".opsy")
		cacheDir := filepath.Join(tempDir, ".opsy/cache")

		assert.DirExists(t, configDir)
		assert.DirExists(t, cacheDir)
	})

	t.Run("handles directory creation errors", func(t *testing.T) {
		tempDir, cleanup := setupTestEnv(t)
		defer cleanup()

		// Create a file where the config directory should be
		err := os.WriteFile(filepath.Join(tempDir, ".opsy"), []byte("test"), 0644)
		require.NoError(t, err)

		manager := New()
		err = manager.LoadConfig()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to create config directory")
	})

	t.Run("handles cache directory creation errors", func(t *testing.T) {
		tempDir, cleanup := setupTestEnv(t)
		defer cleanup()

		// Create config dir but make cache path a file
		require.NoError(t, os.MkdirAll(filepath.Join(tempDir, ".opsy"), 0755))
		err := os.WriteFile(filepath.Join(tempDir, ".opsy/cache"), []byte("test"), 0644)
		require.NoError(t, err)

		manager := New()
		err = manager.LoadConfig()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to create cache directory")
	})
}

// TestValidate verifies configuration validation:
// - Validates complete valid configuration
// - Validates all error conditions
// - Checks all configuration field constraints
func TestValidate(t *testing.T) {
	// Find a shell that exists in the environment
	availableShell := "/bin/sh" // Default to /bin/sh which should exist on most systems
	for _, shell := range []string{"/bin/bash", "/bin/zsh", "/bin/sh"} {
		if _, err := exec.LookPath(shell); err == nil {
			availableShell = shell
			break
		}
	}

	tests := []struct {
		name        string
		config      Config
		expectedErr error
	}{
		{
			name: "valid config",
			config: Config{
				configuration: Configuration{
					Logging: LoggingConfiguration{
						Level: "debug",
						Path:  "test.log",
					},
					Anthropic: AnthropicConfiguration{
						APIKey:      "test-key",
						Model:       "test-model",
						Temperature: 0.5,
						MaxTokens:   100,
					},
					Tools: ToolsConfiguration{
						Timeout: 120,
						Exec: ExecToolConfiguration{
							Timeout: 60,
							Shell:   availableShell,
						},
					},
				},
			},
			expectedErr: nil,
		},
		{
			name: "invalid log level",
			config: Config{
				configuration: Configuration{
					Logging: LoggingConfiguration{
						Level: "invalid",
					},
					Anthropic: AnthropicConfiguration{
						APIKey:      "test-key",
						Temperature: 0.5,
						MaxTokens:   100,
					},
					Tools: ToolsConfiguration{
						Exec: ExecToolConfiguration{
							Shell: availableShell,
						},
					},
				},
			},
			expectedErr: ErrInvalidLogLevel,
		},
		{
			name: "temperature too high",
			config: Config{
				configuration: Configuration{
					Logging: LoggingConfiguration{
						Level: "info",
					},
					Anthropic: AnthropicConfiguration{
						APIKey:      "test-key",
						Temperature: 1.5,
						MaxTokens:   100,
					},
					Tools: ToolsConfiguration{
						Exec: ExecToolConfiguration{
							Shell: availableShell,
						},
					},
				},
			},
			expectedErr: ErrInvalidTemp,
		},
		{
			name: "temperature too low",
			config: Config{
				configuration: Configuration{
					Logging: LoggingConfiguration{
						Level: "info",
					},
					Anthropic: AnthropicConfiguration{
						APIKey:      "test-key",
						Temperature: -0.5,
						MaxTokens:   100,
					},
					Tools: ToolsConfiguration{
						Exec: ExecToolConfiguration{
							Shell: availableShell,
						},
					},
				},
			},
			expectedErr: ErrInvalidTemp,
		},
		{
			name: "invalid max tokens",
			config: Config{
				configuration: Configuration{
					Logging: LoggingConfiguration{
						Level: "info",
					},
					Anthropic: AnthropicConfiguration{
						APIKey:      "test-key",
						Temperature: 0.5,
						MaxTokens:   0,
					},
					Tools: ToolsConfiguration{
						Exec: ExecToolConfiguration{
							Shell: availableShell,
						},
					},
				},
			},
			expectedErr: ErrInvalidMaxTokens,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager := &Config{configuration: tt.config.configuration}
			err := manager.validate()
			if tt.expectedErr != nil {
				assert.ErrorIs(t, err, tt.expectedErr)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestGetLogger_LogLevels verifies log level handling:
// - Supports all valid log levels
// - Properly configures logger with each level
// - Handles invalid log levels
func TestGetLogger_LogLevels(t *testing.T) {
	tempDir, cleanup := setupTestEnv(t)
	defer cleanup()

	tests := []struct {
		name     string
		logLevel string
	}{
		{
			name:     "debug level",
			logLevel: "debug",
		},
		{
			name:     "info level",
			logLevel: "info",
		},
		{
			name:     "warn level",
			logLevel: "warn",
		},
		{
			name:     "error level",
			logLevel: "error",
		},
		{
			name:     "default to info for unknown level",
			logLevel: "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager := New()
			manager.configuration.Logging.Path = filepath.Join(tempDir, "test.log")
			manager.configuration.Logging.Level = tt.logLevel

			logger, err := manager.GetLogger()
			require.NoError(t, err)
			assert.NotNil(t, logger)

			// Clean up the log file
			if err := os.Remove(manager.configuration.Logging.Path); err != nil {
				t.Logf("failed to remove log file: %v", err)
			}
		})
	}
}

// TestLoadConfig_ReadErrors verifies config file reading errors:
// - Handles invalid YAML syntax
// - Handles file permission errors
// - Handles missing files
func TestLoadConfig_ReadErrors(t *testing.T) {
	t.Run("handles invalid config file", func(t *testing.T) {
		tempDir, cleanup := setupTestEnv(t)
		defer cleanup()

		// Set required API key
		if err := os.Setenv("ANTHROPIC_API_KEY", "test-api-key"); err != nil {
			t.Fatalf("failed to set ANTHROPIC_API_KEY environment variable: %v", err)
		}

		// Create an invalid YAML file
		configDir := filepath.Join(tempDir, ".opsy")
		require.NoError(t, os.MkdirAll(configDir, 0755))
		require.NoError(t, os.WriteFile(filepath.Join(configDir, "config.yaml"), []byte("invalid: : : yaml"), 0644))

		manager := New()
		err := manager.LoadConfig()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to read config")
	})
}

// TestLoadConfig_UITheme verifies UI theme configuration:
// - Loads default theme
// - Loads custom theme from config
// - Handles theme from environment variable
// - Properly handles environment variable override
func TestLoadConfig_UITheme(t *testing.T) {
	tempDir, cleanup := setupTestEnv(t)
	defer cleanup()

	tests := []struct {
		name        string
		configData  []byte
		envTheme    string
		expectTheme string
		wantErr     bool
	}{
		{
			name: "default theme",
			configData: []byte(`
anthropic:
  api_key: test-key`),
			expectTheme: "default",
			wantErr:     false,
		},
		{
			name: "custom theme from config",
			configData: []byte(`
anthropic:
  api_key: test-key
ui:
  theme: dark`),
			expectTheme: "dark",
			wantErr:     false,
		},
		{
			name: "theme from environment variable",
			configData: []byte(`
anthropic:
  api_key: test-key`),
			envTheme:    "light",
			expectTheme: "light",
			wantErr:     false,
		},
		{
			name: "environment variable overrides config",
			configData: []byte(`
anthropic:
  api_key: test-key
ui:
  theme: dark`),
			envTheme:    "light",
			expectTheme: "light",
			wantErr:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create config file
			configDir := filepath.Join(tempDir, ".opsy")
			require.NoError(t, os.MkdirAll(configDir, 0755))
			require.NoError(t, os.WriteFile(filepath.Join(configDir, "config.yaml"), tt.configData, 0644))

			// Set environment variable if specified
			if tt.envTheme != "" {
				if err := os.Setenv("OPSY_UI_THEME", tt.envTheme); err != nil {
					t.Fatalf("failed to set OPSY_UI_THEME environment variable: %v", err)
				}
				defer func() {
					if err := os.Unsetenv("OPSY_UI_THEME"); err != nil {
						t.Logf("failed to unset OPSY_UI_THEME environment variable: %v", err)
					}
				}()
			}

			manager := New()
			err := manager.LoadConfig()

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			config := manager.GetConfig()
			assert.Equal(t, tt.expectTheme, string(config.UI.Theme))
		})
	}
}

// TestConfig_Interface verifies interface implementation:
// - Ensures Config implements Configurer interface
func TestConfig_Interface(t *testing.T) {
	// Verify Config implements Configurer interface
	var _ Configurer = (*Config)(nil)
}

// TestLoadConfig_ConfigFilePermissions verifies permission handling:
// - Handles restricted directory permissions
// - Properly reports permission errors
func TestLoadConfig_ConfigFilePermissions(t *testing.T) {
	tempDir, cleanup := setupTestEnv(t)
	defer cleanup()

	// Create config directory with restricted permissions
	configDir := filepath.Join(tempDir, ".opsy")
	require.NoError(t, os.MkdirAll(configDir, 0000))
	//nolint:errcheck
	defer os.Chmod(configDir, 0755) // Restore permissions for cleanup

	manager := New()
	err := manager.LoadConfig()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create directories")
}

// TestGetLogger_ConfiguredLevel verifies logger level configuration:
// - Configures logger with each valid level
// - Creates working logger for each level
// - Handles invalid log levels
func TestGetLogger_ConfiguredLevel(t *testing.T) {
	tempDir, cleanup := setupTestEnv(t)
	defer cleanup()

	manager := New()
	manager.configuration.Logging.Path = filepath.Join(tempDir, "test.log")

	// Test each valid log level
	levels := []string{"debug", "info", "warn", "error"}
	for _, level := range levels {
		t.Run(level, func(t *testing.T) {
			manager.configuration.Logging.Level = level
			logger, err := manager.GetLogger()
			assert.NoError(t, err)
			assert.NotNil(t, logger)
		})
	}
}

// TestLoadConfig_AnthropicValidation verifies Anthropic configuration:
// - Validates model names
// - Handles empty model name
// - Uses default values when appropriate
func TestLoadConfig_AnthropicValidation(t *testing.T) {
	tests := []struct {
		name        string
		config      string
		expectError string
	}{
		{
			name: "valid model name",
			config: `
anthropic:
  api_key: test-key
  model: claude-3-opus
  temperature: 0.7
  max_tokens: 100`,
			expectError: "",
		},
		{
			name: "empty model name",
			config: `
anthropic:
  api_key: test-key
  model: ""
  temperature: 0.7
  max_tokens: 100`,
			expectError: "", // Default model will be used
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tempDir, cleanup := setupTestEnv(t)
			defer cleanup()

			configDir := filepath.Join(tempDir, ".opsy")
			require.NoError(t, os.MkdirAll(configDir, 0755))
			require.NoError(t, os.WriteFile(filepath.Join(configDir, "config.yaml"), []byte(tt.config), 0644))

			manager := New()
			err := manager.LoadConfig()

			if tt.expectError != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectError)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestLoadConfig_ToolsConfiguration verifies tools configuration:
// - Loads default tool timeouts and shell
// - Loads custom tool timeouts and shell from config
// - Handles tool timeouts from environment variables
func TestLoadConfig_ToolsConfiguration(t *testing.T) {
	tempDir, cleanup := setupTestEnv(t)
	defer cleanup()

	// Find a shell that exists in the environment
	availableShell := "/bin/sh" // Default to /bin/sh which should exist on most systems
	for _, shell := range []string{"/bin/bash", "/bin/zsh", "/bin/sh"} {
		if _, err := exec.LookPath(shell); err == nil {
			availableShell = shell
			break
		}
	}

	tests := []struct {
		name           string
		configData     []byte
		envTimeout     string
		envExecTimeout string
		envExecShell   string
		expectTimeout  int64
		expectExec     int64
		expectShell    string
		wantErr        bool
	}{
		{
			name: "default timeouts and shell",
			configData: []byte(`
anthropic:
  api_key: test-key`),
			expectTimeout: 120,
			expectExec:    0,
			expectShell:   "/bin/bash", // This will be overridden by the default in viper
			wantErr:       false,
		},
		{
			name: "custom timeouts and shell from config",
			configData: []byte(fmt.Sprintf(`
anthropic:
  api_key: test-key
tools:
  timeout: 180
  exec:
    timeout: 90
    shell: "%s"`, availableShell)),
			expectTimeout: 180,
			expectExec:    90,
			expectShell:   availableShell,
			wantErr:       false,
		},
		{
			name: "timeouts and shell from environment variables",
			configData: []byte(`
anthropic:
  api_key: test-key`),
			envTimeout:     "240",
			envExecTimeout: "120",
			envExecShell:   availableShell,
			expectTimeout:  240,
			expectExec:     120,
			expectShell:    availableShell,
			wantErr:        false,
		},
		{
			name: "environment variables override config",
			configData: []byte(fmt.Sprintf(`
anthropic:
  api_key: test-key
tools:
  timeout: 180
  exec:
    timeout: 90
    shell: "%s"`, availableShell)),
			envTimeout:     "300",
			envExecTimeout: "150",
			envExecShell:   availableShell,
			expectTimeout:  300,
			expectExec:     150,
			expectShell:    availableShell,
			wantErr:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create config file
			configDir := filepath.Join(tempDir, ".opsy")
			require.NoError(t, os.MkdirAll(configDir, 0755))
			require.NoError(t, os.WriteFile(filepath.Join(configDir, "config.yaml"), tt.configData, 0644))

			// Set environment variables if specified
			if tt.envTimeout != "" {
				if err := os.Setenv("OPSY_TOOLS_TIMEOUT", tt.envTimeout); err != nil {
					t.Fatalf("failed to set OPSY_TOOLS_TIMEOUT environment variable: %v", err)
				}
				defer func() {
					if err := os.Unsetenv("OPSY_TOOLS_TIMEOUT"); err != nil {
						t.Logf("failed to unset OPSY_TOOLS_TIMEOUT environment variable: %v", err)
					}
				}()
			}
			if tt.envExecTimeout != "" {
				if err := os.Setenv("OPSY_TOOLS_EXEC_TIMEOUT", tt.envExecTimeout); err != nil {
					t.Fatalf("failed to set OPSY_TOOLS_EXEC_TIMEOUT environment variable: %v", err)
				}
				defer func() {
					if err := os.Unsetenv("OPSY_TOOLS_EXEC_TIMEOUT"); err != nil {
						t.Logf("failed to unset OPSY_TOOLS_EXEC_TIMEOUT environment variable: %v", err)
					}
				}()
			}
			if tt.envExecShell != "" {
				if err := os.Setenv("OPSY_TOOLS_EXEC_SHELL", tt.envExecShell); err != nil {
					t.Fatalf("failed to set OPSY_TOOLS_EXEC_SHELL environment variable: %v", err)
				}
				defer func() {
					if err := os.Unsetenv("OPSY_TOOLS_EXEC_SHELL"); err != nil {
						t.Logf("failed to unset OPSY_TOOLS_EXEC_SHELL environment variable: %v", err)
					}
				}()
			}

			// For the default test case, we need to set the default shell to an available one
			if tt.name == "default timeouts and shell" {
				if err := os.Setenv("OPSY_TOOLS_EXEC_SHELL", availableShell); err != nil {
					t.Fatalf("failed to set OPSY_TOOLS_EXEC_SHELL environment variable: %v", err)
				}
				defer func() {
					if err := os.Unsetenv("OPSY_TOOLS_EXEC_SHELL"); err != nil {
						t.Logf("failed to unset OPSY_TOOLS_EXEC_SHELL environment variable: %v", err)
					}
				}()
				tt.expectShell = availableShell
			}

			manager := New()
			err := manager.LoadConfig()

			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			config := manager.GetConfig()
			assert.Equal(t, tt.expectTimeout, config.Tools.Timeout)
			assert.Equal(t, tt.expectExec, config.Tools.Exec.Timeout)
			assert.Equal(t, tt.expectShell, config.Tools.Exec.Shell)
		})
	}
}

// TestNewGetConfig_SafeAccess verifies that:
// - GetConfig can be called immediately after New()
// - All configuration fields are accessible without nil panics
// - Default struct values are present
func TestNewGetConfig_SafeAccess(t *testing.T) {
	_, cleanup := setupTestEnv(t)
	defer cleanup()

	// Get config immediately after New() without LoadConfig()
	config := New().GetConfig()

	// Test safe access to all configuration fields
	t.Run("safe access to all fields", func(t *testing.T) {
		// UI Configuration
		assert.Empty(t, config.UI.Theme)

		// Logging Configuration
		assert.Empty(t, config.Logging.Path)
		assert.Empty(t, config.Logging.Level)

		// Anthropic Configuration
		assert.Empty(t, config.Anthropic.APIKey)
		assert.Empty(t, config.Anthropic.Model)
		assert.Zero(t, config.Anthropic.Temperature)
		assert.Zero(t, config.Anthropic.MaxTokens)

		// Tools Configuration
		assert.Zero(t, config.Tools.Timeout)
		assert.Zero(t, config.Tools.Exec.Timeout)
		assert.Empty(t, config.Tools.Exec.Shell)
	})

	// Verify the struct is properly initialized
	t.Run("configuration struct is initialized", func(t *testing.T) {
		// Using reflection to verify the struct is not nil
		assert.NotNil(t, config)
		assert.NotNil(t, config.UI)
		assert.NotNil(t, config.Logging)
		assert.NotNil(t, config.Anthropic)
		assert.NotNil(t, config.Tools)
		assert.NotNil(t, config.Tools.Exec)
	})
}
