#!/bin/zsh
set -eu

uid_value="$(id -u)"
domain="gui/${uid_value}"
launch_agents="${HOME}/Library/LaunchAgents"
runtime_root="${0:A:h:h}/.runtime"

labels=(
  com.islamic-research-ai.postgres
  com.islamic-research-ai.gemma
  com.islamic-research-ai.api
  com.islamic-research-ai.web
  com.islamic-research-ai.tunnel
  com.islamic-research-ai.keep-awake
)

wait_for_url() {
  local url="$1"
  local attempts=30
  while (( attempts > 0 )); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    attempts=$((attempts - 1))
    sleep 1
  done
  return 1
}

start_services() {
  local label
  for label in "${labels[@]}"; do
    launchctl bootstrap "$domain" "${launch_agents}/${label}.plist" 2>/dev/null || true
    launchctl enable "${domain}/${label}"
    launchctl kickstart "${domain}/${label}"
  done
  wait_for_url "http://127.0.0.1:3000"
  wait_for_url "http://127.0.0.1:8000/health"
}

stop_services() {
  local index label
  for (( index=${#labels[@]}; index>=1; index-- )); do
    label="${labels[$index]}"
    launchctl bootout "${domain}/${label}" 2>/dev/null || true
  done
}

service_status() {
  local label details state pid
  for label in "${labels[@]}"; do
    if details="$(launchctl print "${domain}/${label}" 2>/dev/null)"; then
      state="$(printf '%s\n' "$details" | awk -F'= ' '/state =/{print $2; exit}')"
      pid="$(printf '%s\n' "$details" | awk -F'= ' '/pid =/{print $2; exit}')"
      printf '%-42s state=%s pid=%s\n' "$label" "${state:-unknown}" "${pid:-n/a}"
    else
      printf '%-42s state=not-loaded pid=n/a\n' "$label"
    fi
  done
  printf 'api_health='
  curl -fsS --max-time 3 "http://127.0.0.1:8000/health" || printf 'unavailable'
  printf '\npublic_url='
  rg -o 'https://[a-z0-9-]+\.trycloudflare\.com' "${runtime_root}/logs/tunnel-launchd.err.log" 2>/dev/null | tail -1 || printf 'unavailable'
  printf '\n'
}

case "${1:-status}" in
  start)
    start_services
    service_status
    ;;
  restart)
    stop_services
    start_services
    service_status
    ;;
  stop)
    stop_services
    ;;
  status)
    service_status
    ;;
  *)
    printf 'Usage: %s {start|restart|stop|status}\n' "$0" >&2
    exit 2
    ;;
esac
