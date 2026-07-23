# ast-mcp — manage the ast-context-cache MCP server
# Installed by: make install
# Paths are filled in at install time via sed.

set -g __ast_mcp_dir  "__AST_DIR__"
set -g __ast_mcp_ort  "__ORT_LIB__"
set -g __ast_mcp_port 7821
set -g __ast_mcp_dash 7830
set -g __ast_mcp_log "$HOME/.astcache/ast-mcp.log"
set -g __ast_mcp_stopfile "$HOME/.astcache/ast-mcp.supervise-stop"

function __ast_mcp_is_running
    lsof -iTCP:$__ast_mcp_port -sTCP:LISTEN -P 2>/dev/null | grep -q .
end

function __ast_mcp_usage
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
end

function ast-mcp
    if test (count $argv) -eq 0
        __ast_mcp_usage
        return 0
    end

    switch $argv[1]
        case start
            rm -f $__ast_mcp_stopfile
            if __ast_mcp_is_running
                echo "ast-mcp: already running on port $__ast_mcp_port"
                return 0
            end
            echo "Starting ast-mcp..."
            mkdir -p (dirname $__ast_mcp_log)
            cd $__ast_mcp_dir
            set -lx ONNXRUNTIME_LIB $__ast_mcp_ort
            nohup ./ast-mcp > $__ast_mcp_log 2>&1 &
            set -l waited 0
            while test $waited -lt 120
                sleep 3
                set waited (math $waited + 3)
                if __ast_mcp_is_running
                    echo "ast-mcp: started (MCP :$__ast_mcp_port  Dashboard :$__ast_mcp_dash)"
                    return 0
                end
            end
            if __ast_mcp_is_running
                echo "ast-mcp: started (MCP :$__ast_mcp_port  Dashboard :$__ast_mcp_dash)"
            else
                echo "ast-mcp: FAILED — check $__ast_mcp_log"
                return 1
            end

        case start-safe
            rm -f $__ast_mcp_stopfile
            if __ast_mcp_is_running
                echo "ast-mcp: already running on port $__ast_mcp_port"
                return 0
            end
            echo "Starting ast-mcp with 0 embed workers..."
            mkdir -p (dirname $__ast_mcp_log)
            cd $__ast_mcp_dir
            set -lx ONNXRUNTIME_LIB $__ast_mcp_ort
            set -lx AST_EMBED_WORKERS 0
            nohup ./ast-mcp > $__ast_mcp_log 2>&1 &
            sleep 2
            if __ast_mcp_is_running
                echo "ast-mcp: started safe (0 workers; MCP :$__ast_mcp_port  Dashboard :$__ast_mcp_dash)"
            else
                echo "ast-mcp: FAILED — check $__ast_mcp_log"
                return 1
            end

        case supervise
            # Keep-alive: start if not listening, wait for exit/crash, restart with backoff (1s→2s→5s).
            # Stop the loop with Ctrl+C, or run `ast-mcp stop` from another shell.
            mkdir -p (dirname $__ast_mcp_log)
            rm -f $__ast_mcp_stopfile
            echo "ast-mcp: supervising (Ctrl+C here, or 'ast-mcp stop' from another shell)"
            set -l backoff 1
            while true
                if test -f $__ast_mcp_stopfile
                    rm -f $__ast_mcp_stopfile
                    echo "ast-mcp: supervise exiting (stop requested)"
                    return 0
                end
                if __ast_mcp_is_running
                    set -l existing (lsof -t -iTCP:$__ast_mcp_port -sTCP:LISTEN 2>/dev/null | head -1)
                    if test -n "$existing"
                        echo "ast-mcp: already listening (pid $existing); waiting for exit..."
                        while kill -0 $existing 2>/dev/null
                            if test -f $__ast_mcp_stopfile
                                rm -f $__ast_mcp_stopfile
                                echo "ast-mcp: supervise exiting (stop requested)"
                                return 0
                            end
                            sleep 1
                        end
                        if test -f $__ast_mcp_stopfile
                            rm -f $__ast_mcp_stopfile
                            echo "ast-mcp: supervise exiting (stop requested)"
                            return 0
                        end
                        continue
                    end
                end
                echo "ast-mcp: starting under supervise..."
                set -l started_at (date +%s)
                cd $__ast_mcp_dir
                set -lx ONNXRUNTIME_LIB $__ast_mcp_ort
                ./ast-mcp >> $__ast_mcp_log 2>&1
                set -l ec $status
                if test -f $__ast_mcp_stopfile
                    rm -f $__ast_mcp_stopfile
                    echo "ast-mcp: supervise exiting (stop requested, exit $ec)"
                    return 0
                end
                # SIGINT=130, SIGTERM=143, SIGKILL=137 (also raw 15/9)
                if test $ec -eq 130 -o $ec -eq 143 -o $ec -eq 137 -o $ec -eq 15 -o $ec -eq 9
                    echo "ast-mcp: supervise exiting (child stopped, exit $ec)"
                    return 0
                end
                set -l ran (math (date +%s) - $started_at)
                if test $ran -ge 30
                    set backoff 1
                end
                echo "ast-mcp: exited (code $ec); restarting in "$backoff"s..."
                sleep $backoff
                if test $backoff -lt 5
                    if test $backoff -eq 1
                        set backoff 2
                    else
                        set backoff 5
                    end
                end
            end

        case stop
            mkdir -p (dirname $__ast_mcp_log)
            touch $__ast_mcp_stopfile
            set -l pid (lsof -t -iTCP:$__ast_mcp_port -sTCP:LISTEN 2>/dev/null)
            if test -n "$pid"
                kill $pid 2>/dev/null
                set -l i 0
                while test $i -lt 5
                    sleep 1
                    if not __ast_mcp_is_running
                        echo "ast-mcp: stopped (pid $pid)"
                        return 0
                    end
                    set i (math $i + 1)
                end
                kill -9 $pid 2>/dev/null
                echo "ast-mcp: force stopped (pid $pid)"
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
            if echo $resp | grep -qE '"status":"(healthy|starting)"'
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
            __ast_mcp_usage
            return 1
    end
end
