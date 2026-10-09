package server

import (
	"context"
	"net/http"
	"time"
)

// Server wraps the HTTP listener and its rate limiter.
type Server struct {
	httpServer *http.Server
	limiter    *RateLimiter
}

// StartServer listens on addr and serves the shop API until the context ends.
func StartServer(ctx context.Context, addr string, rps int) (*Server, error) {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", handleHealth)
	s := &Server{limiter: NewRateLimiter(rps)}
	s.httpServer = &http.Server{Addr: addr, Handler: s.limiter.Wrap(authMiddleware(mux))}
	go s.httpServer.ListenAndServe()
	go func() {
		<-ctx.Done()
		s.Shutdown(5 * time.Second)
	}()
	return s, nil
}

// Shutdown stops accepting connections and waits up to grace for in-flight requests.
func (s *Server) Shutdown(grace time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), grace)
	defer cancel()
	return s.httpServer.Shutdown(ctx)
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}
