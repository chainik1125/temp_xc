#!/usr/bin/env bash
# supervisor.sh — pod-Claude continuous supervisor (1-hour cadence).
#
# AUTHORIZED BY USER 2026-05-05 for overnight unattended operation.
# The on-pod Claude can auto-execute bash blocks it produces, but
# with the safety rails below.
#
# SAFETY RAILS (enforced in this script):
#   1. Only blocks fenced as EXECUTE_BASH are run.
#   2. Block size capped at 100 lines.
#   3. Forbidden patterns rejected (rm -rf /, sudo, mkfs, dd if=,
#      anything touching /etc, /var, /usr, ~/.ssh/, the env-c6 file).
#   4. All proposed + executed actions appended to supervisor.log
#      with full text + exit code.
#   5. Kill-switch: if /workspace/c6_redteam/PAUSE exists, the
#      supervisor logs and skips intervention (still reports state).
#   6. Max 1 intervention per check-in (no recursive/cascading fixes).
#
# Usage:
#   ./supervisor.sh <orchestrate.sh> <mission.md>

set -e
ORCHESTRATE_SCRIPT="${1:?orchestrate.sh path required}"
MISSION="${2:?mission.md path required}"

. ~/.env-c6
SUP_DIR=/workspace/c6_redteam
ORCH_LOG=$SUP_DIR/orchestrate.log
SUP_LOG=$SUP_DIR/supervisor.log
STATE=$SUP_DIR/state.json
PAUSE_FILE=$SUP_DIR/PAUSE

mkdir -p $SUP_DIR
echo "=== supervisor starting $(date -u) ===" >> $SUP_LOG
echo "  orchestrate: $ORCHESTRATE_SCRIPT" >> $SUP_LOG
echo "  mission:     $MISSION" >> $SUP_LOG
echo "  to pause: 'touch $PAUSE_FILE' (supervisor will skip interventions)" >> $SUP_LOG

chmod +x "$ORCHESTRATE_SCRIPT"
nohup bash "$ORCHESTRATE_SCRIPT" > "$ORCH_LOG" 2>&1 &
ORCH_PID=$!
echo "  orchestrate PID: $ORCH_PID" >> $SUP_LOG

# Forbidden patterns — rejection-list applied to any proposed bash.
FORBIDDEN_REGEX='(rm -rf /([^w]|$))|(rm -rf ~)|(sudo )|(mkfs)|(dd if=)|(:\(\)\{)|(\.ssh)|(env-c6)|(/etc/)|(/usr/bin/)|(/usr/sbin/)|(\bshutdown\b)|(\breboot\b)|(\bhalt\b)|(>.*~/)'

is_forbidden() {
    local script="$1"
    if echo "$script" | grep -qE "$FORBIDDEN_REGEX"; then
        return 0
    fi
    # Length cap: 100 lines.
    if [ "$(wc -l < "$script")" -gt 100 ]; then
        return 0
    fi
    return 1
}

while kill -0 $ORCH_PID 2>/dev/null; do
    sleep 3600

    if ! kill -0 $ORCH_PID 2>/dev/null; then
        break
    fi

    echo "" >> $SUP_LOG
    echo "=== check-in $(date -u) ===" >> $SUP_LOG

    SNAPSHOT=$SUP_DIR/checkin_context.txt
    {
        echo "=== state.json ==="
        cat $STATE 2>/dev/null
        echo
        echo "=== orchestrate.log tail (60 lines, filtered) ==="
        tail -60 $ORCH_LOG | grep -vE "(httpx.INFO HTTP Request|Loading weights:|Fetching .* files:)"
        echo
        echo "=== ps for python processes ==="
        ps -ef | grep python | grep -v grep | head -5
        echo
        echo "=== disk usage ==="
        df -h /workspace / 2>/dev/null | head -3
        echo
        echo "=== nvidia-smi ==="
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 | head -3
        echo
        echo "=== orchestrate alive? ==="
        if kill -0 $ORCH_PID 2>/dev/null; then
            echo "yes (pid $ORCH_PID)"
        else
            echo "NO — orchestrate died"
        fi
        echo
        echo "=== last 50 lines of supervisor.log (your prior decisions) ==="
        tail -50 $SUP_LOG
    } > $SNAPSHOT

    {
        echo "MISSION:"
        cat "$MISSION"
        echo
        echo "CURRENT STATE:"
        cat $SNAPSHOT
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
        echo "  claude call failed; sleeping" >> $SUP_LOG
        continue
    }

    echo "--- claude said ---" >> $SUP_LOG
    cat $SUP_DIR/last_decision.txt >> $SUP_LOG

    # Pause check.
    if [ -f $PAUSE_FILE ]; then
        echo "  PAUSE file present — skipping intervention" >> $SUP_LOG
        continue
    fi

    # Extract EXECUTE_BASH block.
    if grep -q "EXECUTE_BASH:" $SUP_DIR/last_decision.txt; then
        # Extract everything between ```bash (or ```) after EXECUTE_BASH and the closing ```.
        # Tolerant to variants: `EXECUTE_BASH:` then `\`\`\`` then commands then `\`\`\``.
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

echo "=== orchestrate exited $(date -u); supervisor done ===" >> $SUP_LOG
