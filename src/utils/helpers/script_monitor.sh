#!/usr/bin/env bash

INTERVAL=60

while true; do
  printf "\033[2J\033[H"
  echo "==== $(date '+%F %T') ===="

  # CPU + RAM usage (from /proc)
  read -r cpu user nice sys idle iow irq soft steal guest gnice < /proc/stat
  total=$((user+nice+sys+idle+iow+irq+soft+steal))
  busy=$((user+nice+sys+irq+soft+steal))
  cpu_pct=$((100*busy/total))
  mem_line=$(grep -E 'MemTotal|MemAvailable' /proc/meminfo)
  mem_total_kb=$(echo "$mem_line" | awk '/MemTotal/ {print $2}')
  mem_avail_kb=$(echo "$mem_line" | awk '/MemAvailable/ {print $2}')
  mem_used_kb=$((mem_total_kb - mem_avail_kb))
  mem_pct=$((100*mem_used_kb/mem_total_kb))

  echo "CPU: ${cpu_pct}%"
  echo "RAM: ${mem_used_kb}k / ${mem_total_kb}k (${mem_pct}%)"

  echo "----------------------------------------"

  # GPU + python processes (NVIDIA)
  if command -v nvidia-smi >/dev/null 2>&1; then
    declare -A GPU_TOTAL
    declare -A GPU_USED

    echo "GPU summary:"
    while IFS=',' read -r idx uuid total used util; do
      idx=$(echo "$idx" | xargs)
      uuid=$(echo "$uuid" | xargs)
      total=$(echo "$total" | xargs)
      used=$(echo "$used" | xargs)
      util=$(echo "$util" | xargs)
      pct=$((100*used/total))
      echo "  GPU${idx}: ${used} MiB / ${total} MiB (${pct}%) | Util ${util}%"
      GPU_TOTAL["$uuid"]=$total
      GPU_USED["$uuid"]=$used
    done < <(nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)

    # List python GPU processes with per-GPU memory percent
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits \
      | awk -F',' '{
          gsub(/^[ \t]+|[ \t]+$/, "", $1);
          gsub(/^[ \t]+|[ \t]+$/, "", $2);
          gsub(/^[ \t]+|[ \t]+$/, "", $3);
          gsub(/^[ \t]+|[ \t]+$/, "", $4);
          if ($3 ~ /python/) printf "%s,%s,%s,%s\n", $1, $2, $3, $4;
        }' | while IFS=',' read -r uuid pid pname used; do
          total=${GPU_TOTAL["$uuid"]}
          if [ -n "$total" ]; then
            pct=$((100*used/total))
            echo "  GPU PID=${pid} ${pname} ${used} MiB (${pct}%)"
          else
            echo "  GPU PID=${pid} ${pname} ${used} MiB"
          fi
        done
  else
    echo "nvidia-smi not found"
  fi

  echo
  sleep "$INTERVAL"
done
