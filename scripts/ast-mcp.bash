# ast-mcp — manage the ast-context-cache MCP server
# Installed by: make install
# Paths are filled in at install time via sed.
# Works in both bash and zsh.

ast-mcp() {
    local ast_dir="__AST_DIR__"
    local ort_lib="__ORT_LIB__"
    local port=7821
    local dash=7830
    local logfile=/tmp/ast-mcp.log

    _ast_mcp_running() {
        lsof -iTCP:${port} -sTCP:LISTEN -P 2>/dev/null | grep -q .
    }

    case "${1:-}" in
        start)
            if _ast_mcp_running; then
                echo "ast-mcp: already running on port $port"
                return 0
            fi
            echo "Starting ast-mcp..."
            (cd "$ast_dir" && ONNXRUNTIME_LIB="$ort_lib" nohup ./ast-mcp > "$logfile" 2>&1 &)
            sleep 2
            if _ast_mcp_running; then
                echo "ast-mcp: started (MCP :$port  Dashboard :$dash)"
            else
                echo "ast-mcp: FAILED — check $logfile"
                return 1
            fi
            ;;
        stop)
            local pid
            pid=$(lsof -t -iTCP:${port} -sTCP:LISTEN 2>/dev/null)
            if [[ -n "$pid" ]]; then
                kill "$pid" 2>/dev/null
                echo "ast-mcp: stopped (pid $pid)"
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
            if echo "$resp" | grep -q '"healthy"'; then
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
            echo "  start     Start the MCP server"
            echo "  stop      Stop the MCP server"
            echo "  restart   Restart the MCP server"
            echo "  status    Show server status"
            echo "  health    Check server health endpoint"
            echo "  log       Tail the server log"
            echo "  build     Rebuild the binary"
            echo "  dash      Open the dashboard in a browser"
            return 1
            ;;
    esac
}
