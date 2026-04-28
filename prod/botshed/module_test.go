package botshed

import (
	"strings"
	"testing"

	"github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
)

func basenameSet(names ...string) map[string]struct{} {
	m := make(map[string]struct{}, len(names))
	for _, n := range names {
		m[n] = struct{}{}
	}
	return m
}

func TestCountHits(t *testing.T) {
	bn := basenameSet("cd.1posix", "plan9-rc.1", "rc.1plan9", "ls.1", "mwhatis.1")
	cases := []struct {
		name string
		cmd  string
		want int
	}{
		{"empty", "", 0},
		{"whitespace only", "   \t  ", 0},
		{"no token match", "tar xzvf archive.tar.gz", 0},
		{"single match", "ls.1 -la", 1},
		{"multiple matches with shell metachars",
			"cd.1posix $(mktemp -d); plan9-rc.1 b; rc.1plan9 -c", 3},
		{"matches separated by pipes/redirs",
			"plan9-rc.1|rc.1plan9>ls.1", 3},
		{"basename substring shouldn't match",
			"plan9-rc.1.extra cd.1posix.suffix", 0},
		{"backtick and dollar split",
			"`plan9-rc.1`$rc.1plan9", 2},
		{"quotes split tokens",
			"echo \"plan9-rc.1\" 'rc.1plan9'", 2},
		{"ampersand and parens split",
			"(plan9-rc.1 & rc.1plan9)", 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := countHits(tc.cmd, bn)
			if got != tc.want {
				t.Fatalf("countHits(%q) = %d, want %d", tc.cmd, got, tc.want)
			}
		})
	}
}

func TestShouldShed(t *testing.T) {
	bn := basenameSet("cd.1posix", "plan9-rc.1", "rc.1plan9", "ls.1")

	pad := func(cmd string, n int) string {
		if len(cmd) >= n {
			return cmd
		}
		out := cmd + strings.Repeat(" x", (n-len(cmd))/2)
		if len(out) < n {
			out += "y"
		}
		return out
	}

	cases := []struct {
		name     string
		cmd      string
		wantShed bool
	}{
		{"four hits short cmd",
			"plan9-rc.1 rc.1plan9 cd.1posix ls.1", true},
		{"three hits short cmd not enough",
			"plan9-rc.1 rc.1plan9 cd.1posix", false},
		{"two hits just over 250",
			pad("ls.1 plan9-rc.1 ", 251), true},
		{"two hits at 250 boundary",
			pad("ls.1 plan9-rc.1 ", 250), false},
		{"one hit long cmd",
			pad("ls.1 ", 4000), false},
		{"zero hits long cmd",
			pad("tar xzvf archive ", 5000), false},
		{"over 5000 cap not shed",
			pad("plan9-rc.1 rc.1plan9 cd.1posix ls.1 ", 5001), false},
		{"empty not shed",
			"", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			shed, _ := shouldShed(tc.cmd, bn)
			if shed != tc.wantShed {
				t.Fatalf("shouldShed(len=%d) = %v, want %v",
					len(tc.cmd), shed, tc.wantShed)
			}
		})
	}
}

func TestUnmarshalCaddyfile(t *testing.T) {
	t.Run("valid block", func(t *testing.T) {
		input := `botshed {
			db_path /tmp/foo.db
			upstream [::1]:8081
		}`
		d := caddyfile.NewTestDispenser(input)
		var b Botshed
		if err := b.UnmarshalCaddyfile(d); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if b.DBPath != "/tmp/foo.db" {
			t.Errorf("DBPath = %q, want /tmp/foo.db", b.DBPath)
		}
		if b.Upstream != "[::1]:8081" {
			t.Errorf("Upstream = %q, want [::1]:8081", b.Upstream)
		}
	})

	t.Run("unknown option", func(t *testing.T) {
		input := `botshed {
			bogus_key value
		}`
		d := caddyfile.NewTestDispenser(input)
		var b Botshed
		if err := b.UnmarshalCaddyfile(d); err == nil {
			t.Fatal("expected error for unknown option, got nil")
		}
	})
}
