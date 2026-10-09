package server

import (
	"encoding/json"
	"net/http"
)

// Order is the JSON shape returned by the orders endpoint.
type Order struct {
	ID         string `json:"id"`
	TotalCents int    `json:"total_cents"`
}

// RegisterRoutes mounts the order endpoints on mux.
func RegisterRoutes(mux *http.ServeMux, orders func(customer string) []Order) {
	mux.HandleFunc("/orders", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, orders(r.URL.Query().Get("customer")))
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
