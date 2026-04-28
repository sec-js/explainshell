package botshed

import (
	"context"
	"database/sql"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"go.uber.org/zap"

	_ "modernc.org/sqlite"
)

func init() {
	caddy.RegisterModule(&Botshed{})
	httpcaddyfile.RegisterHandlerDirective("botshed", parseCaddyfile)
}

// Seed cmds whose live renders we cache as canned responses.
var seedCmds = []string{
	"ls -la",
	"tar xzvf archive.tar.gz",
	"find . -type f -name '*.py'",
	"git log --oneline -20",
	"cd /tmp && pwd",
	"cat /etc/hosts",
	"echo hello world",
	"grep -rn TODO src/",
}

// Shell metacharacters used for tokenizing cmd values.
const tokenSep = " \t\n\r;|&()<>\"'$`\\"

type Botshed struct {
	DBPath   string `json:"db_path"`
	Upstream string `json:"upstream"`

	basenames atomic.Pointer[map[string]struct{}]
	canned    atomic.Pointer[[][]byte]

	// Counts requests that arrived before both loaders finished, plus a
	// one-shot flag so we log the boot-window total exactly once when we
	// first observe both loaders ready.
	bootPassthroughs atomic.Uint64
	readyLogged      atomic.Bool

	logger *zap.Logger
}

func (*Botshed) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "http.handlers.botshed",
		New: func() caddy.Module { return new(Botshed) },
	}
}

func (b *Botshed) Provision(ctx caddy.Context) error {
	b.logger = ctx.Logger()
	// Caddyfile values like {env.DB_PATH} arrive as raw tokens; expand
	// them once at provision time so the loaders see real paths/addrs.
	repl := caddy.NewReplacer()
	b.DBPath = repl.ReplaceAll(b.DBPath, "")
	b.Upstream = repl.ReplaceAll(b.Upstream, "")
	go b.loadBasenames()
	go b.loadCanned()
	return nil
}

func (b *Botshed) loadBasenames() {
	if b.DBPath == "" {
		b.logger.Warn("botshed: db_path empty, basename detector disabled")
		return
	}
	dsn := fmt.Sprintf("file:%s?mode=ro&immutable=1", b.DBPath)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		b.logger.Error("botshed: open db failed", zap.Error(err))
		return
	}
	defer db.Close()

	rows, err := db.Query("SELECT source FROM manpages")
	if err != nil {
		b.logger.Error("botshed: query manpages failed", zap.Error(err))
		return
	}
	defer rows.Close()

	m := make(map[string]struct{}, 50000)
	for rows.Next() {
		var src string
		if err := rows.Scan(&src); err != nil {
			b.logger.Error("botshed: scan row failed", zap.Error(err))
			return
		}
		base := filepath.Base(src)
		base = strings.TrimSuffix(base, ".gz")
		if base != "" {
			m[base] = struct{}{}
		}
	}
	if err := rows.Err(); err != nil {
		b.logger.Error("botshed: rows iter failed", zap.Error(err))
		return
	}
	b.basenames.Store(&m)
	b.logger.Info("botshed: loaded basenames", zap.Int("count", len(m)))
}

func (b *Botshed) loadCanned() {
	if b.Upstream == "" {
		b.logger.Warn("botshed: upstream empty, canned responses disabled")
		return
	}
	client := &http.Client{Timeout: 10 * time.Second}
	results := make([][]byte, 0, len(seedCmds))
	deadline := time.Now().Add(30 * time.Second)
	delay := 500 * time.Millisecond

	for _, cmd := range seedCmds {
		var body []byte
		for {
			b2, err := fetchOne(client, b.Upstream, cmd)
			if err == nil {
				body = b2
				break
			}
			if time.Now().After(deadline) {
				b.logger.Error("botshed: canned fetch failed, deadline reached",
					zap.String("cmd", cmd), zap.Error(err))
				break
			}
			time.Sleep(delay)
			if delay < 4*time.Second {
				delay *= 2
			}
		}
		if body != nil {
			results = append(results, body)
		}
	}
	if len(results) == 0 {
		b.logger.Error("botshed: no canned responses loaded")
		return
	}
	b.canned.Store(&results)
	b.logger.Info("botshed: loaded canned responses", zap.Int("count", len(results)))
}

func fetchOne(c *http.Client, upstream, cmd string) ([]byte, error) {
	u := fmt.Sprintf("http://%s/explain?cmd=%s", upstream, url.QueryEscape(cmd))
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return body, nil
}

func (b *Botshed) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	if r.Method != http.MethodGet || !strings.HasPrefix(r.URL.Path, "/explain") {
		return next.ServeHTTP(w, r)
	}
	cmd := r.URL.Query().Get("cmd")
	bnPtr := b.basenames.Load()
	cannedPtr := b.canned.Load()
	if bnPtr == nil || cannedPtr == nil {
		b.bootPassthroughs.Add(1)
		return next.ServeHTTP(w, r)
	}
	if b.readyLogged.CompareAndSwap(false, true) {
		b.logger.Info("botshed: ready",
			zap.Uint64("boot_passthroughs", b.bootPassthroughs.Load()),
		)
	}
	bn := *bnPtr
	canned := *cannedPtr

	shed, hits := shouldShed(cmd, bn)
	if !shed {
		return next.ServeHTTP(w, r)
	}

	idx := fnv32(cmd) % uint32(len(canned))
	body := canned[idx]
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "public, max-age=86400")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)

	cmdSnip := cmd
	if len(cmdSnip) > 80 {
		cmdSnip = cmdSnip[:80]
	}
	b.logger.Info("botshed: shed",
		zap.String("client_ip", clientIP(r)),
		zap.String("country", r.Header.Get("CF-IPCountry")),
		zap.String("cmd", cmdSnip),
		zap.Int("hits", hits),
		zap.Uint32("canned_idx", idx),
	)
	return nil
}

// shouldShed is the single source of truth for the detector decision.
// ServeHTTP and the unit tests both call through here so the rule cannot
// drift between production and the tests.
func shouldShed(cmd string, bn map[string]struct{}) (shed bool, hits int) {
	if cmd == "" || len(cmd) > 5000 {
		return false, 0
	}
	hits = countHits(cmd, bn)
	return hits >= 4 || (hits >= 2 && len(cmd) > 250), hits
}

func countHits(cmd string, bn map[string]struct{}) int {
	fields := strings.FieldsFunc(cmd, func(r rune) bool {
		return strings.ContainsRune(tokenSep, r)
	})
	hits := 0
	for _, f := range fields {
		if _, ok := bn[f]; ok {
			hits++
		}
	}
	return hits
}

func fnv32(s string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(s))
	return h.Sum32()
}

func clientIP(r *http.Request) string {
	if v := r.Header.Get("Do-Connecting-Ip"); v != "" {
		return v
	}
	if v := r.Header.Get("CF-Connecting-IP"); v != "" {
		return v
	}
	return r.RemoteAddr
}

func (b *Botshed) UnmarshalCaddyfile(d *caddyfile.Dispenser) error {
	for d.Next() {
		for d.NextBlock(0) {
			switch d.Val() {
			case "db_path":
				if !d.NextArg() {
					return d.ArgErr()
				}
				b.DBPath = d.Val()
			case "upstream":
				if !d.NextArg() {
					return d.ArgErr()
				}
				b.Upstream = d.Val()
			default:
				return d.Errf("unknown botshed option: %s", d.Val())
			}
		}
	}
	return nil
}

func parseCaddyfile(h httpcaddyfile.Helper) (caddyhttp.MiddlewareHandler, error) {
	var b Botshed
	if err := b.UnmarshalCaddyfile(h.Dispenser); err != nil {
		return nil, err
	}
	return &b, nil
}

var (
	_ caddy.Provisioner           = (*Botshed)(nil)
	_ caddyhttp.MiddlewareHandler = (*Botshed)(nil)
	_ caddyfile.Unmarshaler       = (*Botshed)(nil)
)
