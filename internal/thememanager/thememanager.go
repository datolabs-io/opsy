package thememanager

import (
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/datolabs-io/opsy/assets"
	"gopkg.in/yaml.v3"
)

const (
	// ErrThemeNotFound is returned when a theme is not found.
	ErrThemeNotFound = "theme not found"
	// ErrReadingTheme is returned when theme file cannot be read.
	ErrReadingTheme = "failed to read theme file"
	// ErrParsingTheme is returned when theme file cannot be parsed.
	ErrParsingTheme = "failed to parse theme file"
)

const (
	// defaultTheme is the default theme name.
	defaultTheme = "default"
	// themeExtension is the extension for theme files.
	themeExtension = "yaml"
)

// Manager is the interface for the theme manager.
type Manager interface {
	// LoadTheme loads a named theme from the theme manager.
	LoadTheme(name string) error
	// GetTheme returns the current theme.
	GetTheme() *Theme
}

// ThemeManager is the manager for the themes.
type ThemeManager struct {
	logger *slog.Logger
	fs     fs.FS
	dir    string
	theme  *Theme
}

// Option is a function that modifies the theme manager.
type Option func(*ThemeManager)

// New creates a new theme manager.
func New(opts ...Option) *ThemeManager {
	tm := &ThemeManager{
		fs:     assets.Themes,
		dir:    assets.ThemeDir,
		logger: slog.New(slog.DiscardHandler),
	}

	for _, opt := range opts {
		opt(tm)
	}

	tm.logger.WithGroup("config").With("directory", tm.dir).Debug("Theme manager initialized.")

	return tm
}

// WithDirectory sets the directory for the theme manager.
func WithDirectory(dir string) Option {
	return func(tm *ThemeManager) {
		tm.fs = os.DirFS(dir)
		tm.dir = dir
	}
}

// WithLogger sets the logger for the theme manager.
func WithLogger(logger *slog.Logger) Option {
	return func(tm *ThemeManager) {
		tm.logger = logger.With("component", "thememanager")
	}
}

// LoadTheme loads a named theme from the theme manager.
func (tm *ThemeManager) LoadTheme(name string) (err error) {
	if name == "" {
		name = defaultTheme
	}

	var data []byte
	file, err := tm.fs.Open(tm.getFilePath(name))
	if err != nil {
		return fmt.Errorf("%s: %v", ErrThemeNotFound, err)
	}

	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			err = fmt.Errorf("failed to close file: %v", closeErr)
		}
	}()

	data, err = io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("%s: %v", ErrReadingTheme, err)
	}

	if err := yaml.Unmarshal(data, &tm.theme); err != nil {
		return fmt.Errorf("%s: %v", ErrParsingTheme, err)
	}

	tm.logger.WithGroup("theme").With("name", name).Debug("Theme loaded.")

	return nil
}

// GetTheme returns the current theme.
func (tm *ThemeManager) GetTheme() *Theme {
	return tm.theme
}

// getFilePath returns the file path for a given theme name.
func (tm *ThemeManager) getFilePath(name string) string {
	return filepath.Join(tm.dir, fmt.Sprintf("%s.%s", name, themeExtension))
}
