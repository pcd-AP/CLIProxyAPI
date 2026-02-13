// Package util provides utility functions for the CLI Proxy API server.
// It includes helper functions for proxy configuration, HTTP client setup,
// log level management, and other common operations used across the application.
package util

import (
	"context"
	"net"
	"net/http"
	"net/url"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/sdk/config"
	log "github.com/sirupsen/logrus"
	"golang.org/x/net/proxy"
)

// NormalizeProxyURL normalizes a raw proxy string into a valid URL.
// Supported formats:
//   - Already a URL (http://, https://, socks5://, socks4://) → return as-is
//   - host:port (2 parts) → http://host:port
//   - host:port:user:pass (4 parts) → http://user:pass@host:port
//   - Otherwise → prepend http://
func NormalizeProxyURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return raw
	}
	if strings.HasPrefix(raw, "http://") || strings.HasPrefix(raw, "https://") ||
		strings.HasPrefix(raw, "socks5://") || strings.HasPrefix(raw, "socks4://") {
		return raw
	}
	parts := strings.Split(raw, ":")
	switch len(parts) {
	case 2:
		return "http://" + parts[0] + ":" + parts[1]
	case 4:
		return "http://" + parts[2] + ":" + parts[3] + "@" + parts[0] + ":" + parts[1]
	default:
		return "http://" + raw
	}
}

// SetProxy configures the provided HTTP client with proxy settings from the configuration.
// It supports SOCKS5, HTTP, and HTTPS proxies. The function modifies the client's transport
// to route requests through the configured proxy server.
func SetProxy(cfg *config.SDKConfig, httpClient *http.Client) *http.Client {
	var transport *http.Transport
	// Attempt to parse the proxy URL from the configuration.
	proxyURL, errParse := url.Parse(NormalizeProxyURL(cfg.ProxyURL))
	if errParse == nil {
		// Handle different proxy schemes.
		if proxyURL.Scheme == "socks5" {
			// Configure SOCKS5 proxy with optional authentication.
			var proxyAuth *proxy.Auth
			if proxyURL.User != nil {
				username := proxyURL.User.Username()
				password, _ := proxyURL.User.Password()
				proxyAuth = &proxy.Auth{User: username, Password: password}
			}
			dialer, errSOCKS5 := proxy.SOCKS5("tcp", proxyURL.Host, proxyAuth, proxy.Direct)
			if errSOCKS5 != nil {
				log.Errorf("create SOCKS5 dialer failed: %v", errSOCKS5)
				return httpClient
			}
			// Set up a custom transport using the SOCKS5 dialer.
			transport = &http.Transport{
				DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
					return dialer.Dial(network, addr)
				},
			}
		} else if proxyURL.Scheme == "http" || proxyURL.Scheme == "https" {
			// Configure HTTP or HTTPS proxy.
			transport = &http.Transport{Proxy: http.ProxyURL(proxyURL)}
		}
	}
	// If a new transport was created, apply it to the HTTP client.
	if transport != nil {
		httpClient.Transport = transport
	}
	return httpClient
}
