#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_FRONTEND_REPO="$(cd -- "${BACKEND_REPO}/../Dominic/chatbot-ui" 2>/dev/null && pwd || true)"

FRONTEND_REPO="${DEFAULT_FRONTEND_REPO}"
BRANCH="main"
TARGET="all"
SKIP_PULL="0"
NO_BUILD="0"
ENV_FILE="${BACKEND_REPO}/.env.ec2"
COMPOSE_FILE="${BACKEND_REPO}/deploy/docker-compose.ec2.yml"

usage() {
  cat <<'EOF'
Usage: ./scripts/deploy_ec2.sh [options]

Options:
  --branch <name>           Git branch to pull from for both repos. Default: main
  --backend-repo <path>     Absolute path to DominicBE on EC2.
  --frontend-repo <path>    Absolute path to Dominic/chatbot-ui on EC2.
  --env-file <path>         Path to .env.ec2. Default: <backend>/.env.ec2
  --target <all|backend|frontend>
                            Service(s) to rebuild/restart. Default: all
  --skip-pull               Skip git pull and only rebuild/restart containers.
  --no-build                Recreate/start containers without rebuilding images.
  -h, --help                Show this help text.

Examples:
  ./scripts/deploy_ec2.sh
  ./scripts/deploy_ec2.sh --target backend
  ./scripts/deploy_ec2.sh --branch main --skip-pull
EOF
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --backend-repo)
      BACKEND_REPO="$2"
      shift 2
      ;;
    --frontend-repo)
      FRONTEND_REPO="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --skip-pull)
      SKIP_PULL="1"
      shift
      ;;
    --no-build)
      NO_BUILD="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "$TARGET" in
  all|backend|frontend)
    ;;
  *)
    echo "Invalid --target value: $TARGET" >&2
    usage
    exit 1
    ;;
esac

require_command git
require_command docker
require_command curl

if [[ ! -d "$BACKEND_REPO/.git" ]]; then
  echo "Backend repo not found: $BACKEND_REPO" >&2
  exit 1
fi

if [[ "$TARGET" == "all" || "$TARGET" == "frontend" ]]; then
  if [[ -z "$FRONTEND_REPO" || ! -d "$FRONTEND_REPO/.git" ]]; then
    echo "Frontend repo not found. Pass --frontend-repo <path>." >&2
    exit 1
  fi
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

echo "=== Dominic EC2 deploy ==="
echo "Backend repo : $BACKEND_REPO"
echo "Frontend repo: ${FRONTEND_REPO:-<not used>}"
echo "Branch       : $BRANCH"
echo "Target       : $TARGET"
echo "Env file     : $ENV_FILE"
echo "Compose file : $COMPOSE_FILE"

if [[ "$SKIP_PULL" != "1" ]]; then
  echo "=== Pulling backend repo ==="
  git -C "$BACKEND_REPO" pull --ff-only origin "$BRANCH"

  if [[ "$TARGET" == "all" || "$TARGET" == "frontend" ]]; then
    echo "=== Pulling frontend repo ==="
    git -C "$FRONTEND_REPO" pull --ff-only origin "$BRANCH"
  fi
fi

COMPOSE_ARGS=(--env-file "$ENV_FILE" -f "$COMPOSE_FILE")
UP_ARGS=(up -d)

if [[ "$NO_BUILD" != "1" ]]; then
  UP_ARGS+=(--build)
fi

case "$TARGET" in
  all)
    ;;
  backend)
    UP_ARGS+=(backend)
    ;;
  frontend)
    UP_ARGS+=(frontend)
    ;;
esac

echo "=== Applying Docker Compose changes ==="
(
  cd "$BACKEND_REPO"
  docker compose "${COMPOSE_ARGS[@]}" "${UP_ARGS[@]}"
)

echo "=== Container status ==="
(
  cd "$BACKEND_REPO"
  docker compose "${COMPOSE_ARGS[@]}" ps
)

if [[ "$TARGET" == "all" || "$TARGET" == "backend" ]]; then
  echo "=== Backend health ==="
  curl --fail --silent --show-error http://127.0.0.1:8000/health
  echo
fi

if [[ "$TARGET" == "all" || "$TARGET" == "frontend" ]]; then
  echo "=== Frontend health ==="
  curl --fail --silent --show-error http://127.0.0.1:8080/healthz
  echo
fi

echo "=== Deploy completed ==="