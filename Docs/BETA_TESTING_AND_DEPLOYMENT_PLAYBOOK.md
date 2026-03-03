# Yaga Beta Testing and Deployment Playbook

## 1. Purpose

This playbook defines how to:

1. beta test the current Yaga codebase end-to-end
2. evaluate release readiness with measurable exit criteria
3. deploy the product safely in future staging and production environments

Audience: product owners, QA, DevOps, engineering, and AI/code agents assisting the project.

---

## 2. Product Scope (Current)

Yaga is a community help platform with:

- account creation/login/logout
- profile management and visibility controls
- support board (help request/offer lifecycle)
- follows/followers network actions
- direct messaging chat
- appreciation/reward actions
- transformations and leaderboard
- audit ledger for help lifecycle + optional blockchain transaction anchoring
- optional MetaMask + smart-contract sync in the Support page

Primary local URLs:

- frontend: `http://localhost:8080`
- backend API: `http://localhost:3000/api/v1`
- MongoDB: `mongodb://localhost:27017`

---

## 3. Beta Testing Objectives

### Functional objectives

1. all critical user journeys are stable across fresh and returning sessions
2. data integrity is preserved across help lifecycle transitions
3. auth/authorization rules are enforced consistently
4. social graph and chat routes are reliable
5. ledger records capture backend and wallet-originated events correctly

### Non-functional objectives

1. acceptable behavior on modern desktop/mobile browsers
2. basic resilience under concurrent users
3. logging and diagnostics are sufficient for issue triage
4. no critical security regressions (auth bypass, mass-assignment, CORS abuse)

---

## 4. Beta Environment Setup

## 4.1 Prerequisites

- Docker Engine + Docker Compose
- macOS/Linux shell
- curl + ruby (for quick JSON extraction in command-line tests)
- Chrome/Edge (for MetaMask testing)

## 4.2 Startup

From repo root:

```bash
cd /Users/jkarnati/workspaces/yaga-dedicated
docker-compose up -d --build
```

Health checks:

```bash
curl -s http://localhost:8080 | head -n 3
curl -s http://localhost:3000/api/v1/helps/prioritized
```

Logs (live):

```bash
docker-compose logs -f backend frontend mongodb
```

---

## 5. Test Accounts and Data

Create test users with API:

```bash
EMAIL="beta$(date +%s)@yaga.local"
PASS="password123"

curl -s -X POST http://localhost:3000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\",\"password_confirmation\":\"$PASS\",\"first_name\":\"Beta\",\"last_name\":\"User\",\"user_type\":\"Individual\"}"
```

Get token:

```bash
TOKEN=$(curl -s -X POST http://localhost:3000/api/v1/auth/signin \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\"}" \
  | ruby -rjson -e 'puts JSON.parse(STDIN.read)["token"]')
```

---

## 6. Functional Test Matrix

## 6.1 Auth

1. Create Account with valid profile fields
2. Login with valid credentials
3. Invalid login gives `401`
4. Logout invalidates current session
5. Protected endpoints reject missing/invalid token

API checks:

```bash
curl -s http://localhost:3000/api/v1/auth/me -H "Authorization: Bearer $TOKEN"
```

## 6.2 Profile

1. View own profile after login
2. Edit profile values and persist
3. Public profile access honors visibility rules
4. Follow/unfollow reflects counts and relationship status

## 6.3 Support Board and Help Lifecycle

1. Guest can post anonymous help request
2. Authenticated user can post request/offer
3. Another user can take open help
4. Participants can move status to `in_progress`, `completed`, `cancelled`
5. Non-participants cannot mutate lifecycle actions
6. Priority ordering is present in support list

API checks:

```bash
curl -s http://localhost:3000/api/v1/helps/prioritized
curl -s -X POST http://localhost:3000/api/v1/helps/<HELP_ID>/take -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{}'
```

## 6.4 Ledger and Audit Trails

1. Creating help auto-creates backend ledger event (`action=create`)
2. Lifecycle transitions auto-log actions (`take`, `complete`, etc.)
3. Support UI can load audit events per help
4. Manual wallet/indexer event posting works for participants
5. Non-participant posting to ledger returns `403`

API checks:

```bash
curl -s http://localhost:3000/api/v1/helps/<HELP_ID>/ledger_events
curl -s -X POST http://localhost:3000/api/v1/helps/<HELP_ID>/ledger_events \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"ledger_event":{"action":"anchor","source_type":"wallet","chain_state":"submitted","tx_hash":"0x..."}}'
```

## 6.5 Chat

1. Create chat with another profile
2. Send and fetch messages
3. Message history is ordered and scoped correctly

## 6.6 Appreciation/Transformations/Leaderboard

1. Appreciation can be sent only for completed helps
2. Transformations include completed relevant helps
3. Leaderboard ranks according to contribution logic

## 6.7 Wallet + Smart Contract Sync (Optional Beta Track)

Precondition: deploy `contracts/HelpRegistry.sol` and set contract address in Support page.

1. Open `/#/support`, login, configure chain ID + contract
2. Connect MetaMask
3. Enable auto chain sync
4. Create help; verify tx confirmation in wallet
5. Verify ledger shows wallet event with tx hash and chain metadata
6. Run `take/complete/cancel`; verify corresponding on-chain tx and ledger entries

Expected behavior if wallet/contract unavailable:

- backend flow still succeeds
- warning message indicates chain sync was skipped/failed

---

## 7. Regression Checklist (Every Beta Build)

1. Home/Profile routing and auth-aware nav links
2. Support page rendering without raw Angular template interpolation text
3. No console injector errors in browser
4. `docker-compose up -d --build` succeeds from clean machine
5. Fresh login session after browser cache clear

---

## 8. Defect Triage and Severity

- Sev-1: data loss, auth bypass, core flow unavailable
- Sev-2: major functional break with workaround
- Sev-3: minor functional/UI issue
- Sev-4: cosmetic/low priority

For every defect capture:

1. exact route/API
2. request payload and response
3. user role and token context
4. backend/frontend log snippets
5. reproducibility steps

---

## 9. Beta Exit Criteria

1. no open Sev-1 issues
2. Sev-2 issues accepted with explicit product signoff
3. critical flows pass twice consecutively on clean environment
4. deployment playbook dry-run completed
5. rollback procedure validated

---

## 10. Future Product Deployment Plan

## 10.1 Target architecture (recommended)

1. Frontend: static hosting + CDN (CloudFront/S3 or equivalent)
2. Backend API: containerized Rails app on ECS/Kubernetes
3. Database: managed MongoDB (Atlas or managed cluster)
4. Background workers:
   - ledger reconciler
   - chain indexer listener (future)
5. Observability:
   - centralized logs
   - metrics + alerts
   - uptime checks
6. Secrets manager for keys/tokens

## 10.2 Environments

- `dev`: rapid iteration
- `staging`: production-like data model/integration validation
- `prod`: customer-facing, strict controls

## 10.3 CI/CD pipeline

1. lint/static checks
2. unit/integration tests
3. container image build + vulnerability scan
4. push immutable image tag
5. deploy to staging
6. run smoke suite
7. manual approval gate
8. production deployment (blue/green preferred)
9. post-deploy health checks

## 10.4 Configuration and secrets (minimum)

- `RAILS_ENV`
- `MONGODB_URI`
- token/session-related secrets
- CORS allowlist (tightened from `*` for production)
- chain config:
  - chain IDs
  - contract addresses
  - indexer RPC endpoints (future)

## 10.5 Security hardening for production

1. restrict CORS origins and headers
2. enforce TLS everywhere
3. add rate limiting on auth and chat endpoints
4. rotate secrets periodically
5. review authorization coverage for all participant-only actions
6. validate and sanitize all user-input fields

## 10.6 Data protection and resilience

1. scheduled Mongo backups + restore drills
2. retention policy for logs and ledger records
3. disaster recovery runbook with RPO/RTO targets

## 10.7 Rollback strategy

1. retain previous container image and config
2. database backward-compatible rollout policy
3. one-command rollback procedure documented and tested in staging

---

## 11. Post-Deployment Monitoring

Monitor continuously:

- API latency and error rate
- auth failures / suspicious spikes
- help lifecycle transition failure rates
- chat send/read latency
- ledger write failure rates
- smart-contract sync failure rates

Alert immediately for:

- sustained `5xx` errors
- unauthorized access anomalies
- ledger/indexer drift beyond threshold

---

## 12. Suggested Next Automation

1. add automated API regression test suite (Postman/Newman or RSpec request tests)
2. add browser E2E suite for primary journeys
3. add indexer worker for contract event ingestion
4. add reconciliation dashboard for backend vs on-chain state

