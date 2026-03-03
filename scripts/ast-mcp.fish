# ast-mcp — manage the ast-context-cache MCP server
# Installed by: make install
# Paths are filled in at install time via sed.

set -g __ast_mcp_dir  "__AST_DIR__"
set -g __ast_mcp_ort  "__ORT_LIB__"
set -g __ast_mcp_port 7821
set -g __ast_mcp_dash 7830
set -g __ast_mcp_log  /tmp/ast-mcp.log

function __ast_mcp_is_running
    lsof -iTCP:$__ast_mcp_port -sTCP:LISTEN -P 2>/dev/null | grep -q .
end

function ast-mcp
    if test (count $argv) -eq 0
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
        return 0
    end

    switch $argv[1]
        case start
            if __ast_mcp_is_running
                echo "ast-mcp: already running on port $__ast_mcp_port"
                return 0
            end
            echo "Starting ast-mcp..."
            cd $__ast_mcp_dir
            set -lx ONNXRUNTIME_LIB $__ast_mcp_ort
            nohup ./ast-mcp > $__ast_mcp_log 2>&1 &
            sleep 2
            if __ast_mcp_is_running
                echo "ast-mcp: started (MCP :$__ast_mcp_port  Dashboard :$__ast_mcp_dash)"
            else
                echo "ast-mcp: FAILED — check $__ast_mcp_log"
                return 1
            end

        case stop
            set -l pid (lsof -t -iTCP:$__ast_mcp_port -sTCP:LISTEN 2>/dev/null)
            if test -n "$pid"
                kill $pid 2>/dev/null
                echo "ast-mcp: stopped (pid $pid)"
            else
                echo "ast-mcp: not running"
            end

        case restart
            ast-mcp stop
            sleep 1
            ast-mcp start

        case status
            if __ast_mcp_is_running
                set -l pid (lsof -t -iTCP:$__ast_mcp_port -sTCP:LISTEN 2>/dev/null)
                set_color green
                echo "ast-mcp: running (pid $pid)"
                set_color normal
                echo "  MCP:       http://localhost:$__ast_mcp_port/mcp"
                echo "  Dashboard: http://localhost:$__ast_mcp_dash"
            else
                set_color red
                echo "ast-mcp: not running"
                set_color normal
            end

        case health
            if not __ast_mcp_is_running
                set_color red; echo "ast-mcp: not running"; set_color normal
                return 1
            end
            set -l resp (curl -s -m 2 http://localhost:$__ast_mcp_port/health 2>/dev/null)
            if echo $resp | grep -q '"healthy"'
                set_color green; echo "ast-mcp: healthy"; set_color normal
                echo "  $resp"
            else
                set_color yellow; echo "ast-mcp: unhealthy"; set_color normal
                echo "  $resp"
                return 1
            end

        case log
            if test -f $__ast_mcp_log
                tail -f $__ast_mcp_log
            else
                echo "No log file at $__ast_mcp_log"
            end

        case build
            echo "Rebuilding ast-mcp..."
            make -C $__ast_mcp_dir build
            echo "Done."

        case dash
            open "http://localhost:$__ast_mcp_dash" 2>/dev/null \
                || xdg-open "http://localhost:$__ast_mcp_dash" 2>/dev/null \
                || echo "http://localhost:$__ast_mcp_dash"

        case '*'
            echo "Unknown command: $argv[1]"
            ast-mcp
            return 1
    end
end
