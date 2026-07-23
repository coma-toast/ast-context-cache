# ast-mcp — manage the ast-context-cache MCP server
# Installed by: make install
# Paths are filled in at install time via sed.
# Works in both bash and zsh.

ast-mcp() {
    local ast_dir="__AST_DIR__"
    local ort_lib="__ORT_LIB__"
    local port=7821
    local dash=7830
    local logfile="${HOME:+$HOME/.astcache/ast-mcp.log}"
    logfile="${logfile:-.astcache/ast-mcp.log}"

    local stopfile
    stopfile="$(dirname "$logfile")/ast-mcp.supervise-stop"

    _ast_mcp_running() {
        lsof -iTCP:${port} -sTCP:LISTEN -P 2>/dev/null | grep -q .
    }

    case "${1:-}" in
        start)
            rm -f "$stopfile"
            if _ast_mcp_running; then
                echo "ast-mcp: already running on port $port"
                return 0
            fi
            echo "Starting ast-mcp..."
            mkdir -p "$(dirname "$logfile")"
            (cd "$ast_dir" && ONNXRUNTIME_LIB="$ort_lib" nohup ./ast-mcp > "$logfile" 2>&1 &)
            local waited=0
            while [[ $waited -lt 120 ]]; do
                sleep 3
                waited=$((waited + 3))
                if _ast_mcp_running; then
                    echo "ast-mcp: started (MCP :$port  Dashboard :$dash)"
                    return 0
                fi
            done
            if _ast_mcp_running; then
                echo "ast-mcp: started (MCP :$port  Dashboard :$dash)"
            else
                echo "ast-mcp: FAILED — check $logfile"
                return 1
            fi
            ;;
        start-safe)
            rm -f "$stopfile"
            if _ast_mcp_running; then
                echo "ast-mcp: already running on port $port"
                return 0
            fi
            echo "Starting ast-mcp with 0 embed workers..."
            mkdir -p "$(dirname "$logfile")"
            (cd "$ast_dir" && AST_EMBED_WORKERS=0 ONNXRUNTIME_LIB="$ort_lib" nohup ./ast-mcp > "$logfile" 2>&1 &)
            sleep 2
            if _ast_mcp_running; then
                echo "ast-mcp: started safe (0 workers; MCP :$port  Dashboard :$dash)"
            else
                echo "ast-mcp: FAILED — check $logfile"
                return 1
            fi
            ;;
        supervise)
            # Keep-alive: start if not listening, wait for exit/crash, restart with backoff (1s→2s→5s).
            # Stop the loop with Ctrl+C, or run `ast-mcp stop` from another shell (kills child; supervise exits).
            mkdir -p "$(dirname "$logfile")"
            rm -f "$stopfile"
            echo "ast-mcp: supervising (Ctrl+C here, or 'ast-mcp stop' from another shell)"
            local backoff=1
            local child_pid=""
            local started_at=0
            _ast_mcp_supervise_cleanup() {
                touch "$stopfile"
                if [[ -n "${child_pid:-}" ]] && kill -0 "$child_pid" 2>/dev/null; then
                    kill "$child_pid" 2>/dev/null
                    wait "$child_pid" 2>/dev/null || true
                fi
                echo "ast-mcp: supervise stopped"
                exit 0
            }
            trap '_ast_mcp_supervise_cleanup' INT TERM
            while true; do
                if [[ -f "$stopfile" ]]; then
                    rm -f "$stopfile"
                    echo "ast-mcp: supervise exiting (stop requested)"
                    return 0
                fi
                if _ast_mcp_running; then
                    local existing
                    existing=$(lsof -t -iTCP:${port} -sTCP:LISTEN 2>/dev/null | head -1)
                    if [[ -n "$existing" ]]; then
                        echo "ast-mcp: already listening (pid $existing); waiting for exit..."
                        while kill -0 "$existing" 2>/dev/null; do
                            if [[ -f "$stopfile" ]]; then
                                rm -f "$stopfile"
                                echo "ast-mcp: supervise exiting (stop requested)"
                                return 0
                            fi
                            sleep 1
                        done
                        if [[ -f "$stopfile" ]]; then
                            rm -f "$stopfile"
                            echo "ast-mcp: supervise exiting (stop requested)"
                            return 0
                        fi
                        continue
                    fi
                fi
                echo "ast-mcp: starting under supervise..."
                (cd "$ast_dir" && ONNXRUNTIME_LIB="$ort_lib" ./ast-mcp >> "$logfile" 2>&1) &
                child_pid=$!
                started_at=$(date +%s)
                wait "$child_pid"
                local ec=$?
                child_pid=""
                if [[ -f "$stopfile" ]]; then
                    rm -f "$stopfile"
                    echo "ast-mcp: supervise exiting (stop requested, exit $ec)"
                    return 0
                fi
                # SIGINT=130, SIGTERM=143, SIGKILL=137 (also raw 15/9 on some shells)
                if [[ $ec -eq 130 || $ec -eq 143 || $ec -eq 137 || $ec -eq 15 || $ec -eq 9 ]]; then
                    echo "ast-mcp: supervise exiting (child stopped, exit $ec)"
                    return 0
                fi
                local ran=$(( $(date +%s) - started_at ))
                if [[ $ran -ge 30 ]]; then
                    backoff=1
                fi
                echo "ast-mcp: exited (code $ec); restarting in ${backoff}s..."
                sleep "$backoff"
                if [[ $backoff -lt 5 ]]; then
                    if [[ $backoff -eq 1 ]]; then
                        backoff=2
                    else
                        backoff=5
                    fi
                fi
            done
            ;;
        stop)
            mkdir -p "$(dirname "$logfile")"
            touch "$stopfile"
            local pid
            pid=$(lsof -t -iTCP:${port} -sTCP:LISTEN 2>/dev/null)
            if [[ -n "$pid" ]]; then
                kill "$pid" 2>/dev/null
                local i
                for i in 1 2 3 4 5; do
                    sleep 1
                    _ast_mcp_running || break
                done
                if _ast_mcp_running; then
                    kill -9 "$pid" 2>/dev/null
                    echo "ast-mcp: force stopped (pid $pid)"
                else
                    echo "ast-mcp: stopped (pid $pid)"
                fi
            else
                echo "ast-mcp: not running"
            fi
            ;;
        restart)
            ast-mcp stop
            sleep 1
            ast-mcp start
            ;;
        status)
            if _ast_mcp_running; then
                local pid
                pid=$(lsof -t -iTCP:${port} -sTCP:LISTEN 2>/dev/null)
                echo "ast-mcp: running (pid $pid)"
                echo "  MCP:       http://localhost:$port/mcp"
                echo "  Dashboard: http://localhost:$dash"
            else
                echo "ast-mcp: not running"
            fi
            ;;
        health)
            if ! _ast_mcp_running; then
                echo "ast-mcp: not running"
                return 1
            fi
            local resp
            resp=$(curl -s -m 2 "http://localhost:$port/health" 2>/dev/null)
            if echo "$resp" | grep -qE '"status":"(healthy|starting)"'; then
                echo "ast-mcp: healthy"
                echo "  $resp"
            else
                echo "ast-mcp: unhealthy"
                echo "  $resp"
                return 1
            fi
            ;;
        log)
            if [[ -f "$logfile" ]]; then
                tail -f "$logfile"
            else
                echo "No log file at $logfile"
            fi
            ;;
        build)
            echo "Rebuilding ast-mcp..."
            make -C "$ast_dir" build
            echo "Done."
            ;;
        dash)
            if command -v open &>/dev/null; then
                open "http://localhost:$dash"
            elif command -v xdg-open &>/dev/null; then
                xdg-open "http://localhost:$dash"
            else
                echo "http://localhost:$dash"
            fi
            ;;
        *)
            echo "Usage: ast-mcp <command>"
            echo ""
            echo "Commands:"
            echo "  start       Start the MCP server"
            echo "  start-safe  Start with 0 embed workers (crash recovery)"
            echo "  supervise   Keep-alive loop (restart on crash; Ctrl+C or stop to exit)"
            echo "  stop        Stop the MCP server (also ends a supervise loop)"
            echo "  restart     Restart the MCP server"
            echo "  status      Show server status"
            echo "  health      Check server health endpoint"
            echo "  log         Tail the server log"
            echo "  build       Rebuild the binary"
            echo "  dash        Open the dashboard in a browser"
            return 1
            ;;
    esac
}
