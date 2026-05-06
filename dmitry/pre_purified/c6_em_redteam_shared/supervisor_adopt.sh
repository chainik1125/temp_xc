#!/usr/bin/env bash
# supervisor_adopt.sh — like supervisor.sh, but adopts an EXISTING
# orchestrate process by PID instead of spawning a new one.
#
# Used to hot-swap a broken supervisor without disturbing
# in-progress training.
#
# Usage:
#   ./supervisor_adopt.sh <ORCH_PID> <mission.md>

set -e
ORCH_PID="${1:?ORCH_PID required}"
MISSION="${2:?mission.md path required}"

. ~/.env-c6
SUP_DIR=/workspace/c6_redteam
ORCH_LOG=$SUP_DIR/orchestrate.log
SUP_LOG=$SUP_DIR/supervisor.log
STATE=$SUP_DIR/state.json
PAUSE_FILE=$SUP_DIR/PAUSE

if ! kill -0 $ORCH_PID 2>/dev/null; then
    echo "ERROR: orchestrate PID $ORCH_PID is not alive" >> $SUP_LOG
    exit 1
fi

mkdir -p $SUP_DIR
echo "=== supervisor (adopt) starting $(date -u) ===" >> $SUP_LOG
echo "  adopting orchestrate PID: $ORCH_PID" >> $SUP_LOG
echo "  mission: $MISSION" >> $SUP_LOG

FORBIDDEN_REGEX='(rm -rf /([^w]|$))|(rm -rf ~)|(sudo )|(mkfs)|(dd if=)|(:\(\)\{)|(\.ssh)|(env-c6)|(/etc/)|(/usr/bin/)|(/usr/sbin/)|(\bshutdown\b)|(\breboot\b)|(\bhalt\b)|(>.*~/)'

is_forbidden() {
    local script="$1"
    if echo "$script" | grep -qE "$FORBIDDEN_REGEX"; then return 0; fi
    if [ "$(wc -l < "$script")" -gt 100 ]; then return 0; fi
    return 1
}

while kill -0 $ORCH_PID 2>/dev/null; do
    sleep 3600
    if ! kill -0 $ORCH_PID 2>/dev/null; then break; fi

    echo "" >> $SUP_LOG
    echo "=== check-in (adopt) $(date -u) ===" >> $SUP_LOG

    SNAPSHOT=$SUP_DIR/checkin_context.txt
    {
        echo "=== state.json ==="; cat $STATE 2>/dev/null
        echo
        echo "=== orchestrate.log tail (60 lines, filtered) ==="
        tail -60 $ORCH_LOG | grep -vE "(httpx.INFO HTTP Request|Loading weights:|Fetching .* files:)"
        echo
        echo "=== ps for python processes ==="
        ps -ef | grep python | grep -v grep | head -5
        echo
        echo "=== disk usage ==="; df -h /workspace / 2>/dev/null | head -3
        echo
        echo "=== nvidia-smi ==="
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 | head -3
        echo
        echo "=== orchestrate alive? ==="
        if kill -0 $ORCH_PID 2>/dev/null; then echo "yes (pid $ORCH_PID)"; else echo "NO"; fi
        echo
        echo "=== last 50 lines of supervisor.log ==="; tail -50 $SUP_LOG
    } > $SNAPSHOT

    {
        echo "MISSION:"; cat "$MISSION"
        echo
        echo "CURRENT STATE:"; cat $SNAPSHOT
        echo
        echo "Decide one of:"
        echo "  A) all good, no action needed (output exactly: NOACTION)"
        echo "  B) intervene (output a fenced bash block: EXECUTE_BASH:"
        echo "     followed by ... in triple-backtick fences)"
        echo "  C) escalate to user (output: ESCALATE: followed by a 1-paragraph summary)"
        echo
        echo "Be conservative — prefer NOACTION if uncertain."
        echo "Forbidden: rm -rf /, sudo, mkfs, dd, anything touching ~/.ssh, ~/.env-c6,"
        echo "/etc, /usr/bin, /usr/sbin, shutdown/reboot/halt commands."
        echo "Bash block max 100 lines."
    } | claude --print > $SUP_DIR/last_decision.txt 2>&1 || {
        echo "  claude call failed (exit=$?); sleeping" >> $SUP_LOG
        continue
    }

    echo "--- claude said ---" >> $SUP_LOG
    cat $SUP_DIR/last_decision.txt >> $SUP_LOG

    if [ -f $PAUSE_FILE ]; then
        echo "  PAUSE file present — skipping intervention" >> $SUP_LOG
        continue
    fi

    if grep -q "EXECUTE_BASH:" $SUP_DIR/last_decision.txt; then
        awk '
            /EXECUTE_BASH:/ {found=1; next}
            found && /^```/ {if(in_block){exit} else {in_block=1; next}}
            in_block {print}
        ' $SUP_DIR/last_decision.txt > $SUP_DIR/last_intervention.sh

        if [ ! -s $SUP_DIR/last_intervention.sh ]; then
            echo "  EXECUTE_BASH tag found but no extractable block — skipping" >> $SUP_LOG
            continue
        fi

        if is_forbidden $SUP_DIR/last_intervention.sh; then
            echo "  intervention REJECTED (forbidden pattern or >100 lines):" >> $SUP_LOG
            cat $SUP_DIR/last_intervention.sh >> $SUP_LOG
            continue
        fi

        echo "  executing claude's bash block:" >> $SUP_LOG
        cat $SUP_DIR/last_intervention.sh >> $SUP_LOG
        echo "  --- output ---" >> $SUP_LOG
        bash $SUP_DIR/last_intervention.sh >> $SUP_LOG 2>&1
        echo "  intervention exit=$?" >> $SUP_LOG
    fi
done

echo "=== orchestrate (adopted) exited $(date -u); supervisor done ===" >> $SUP_LOG
