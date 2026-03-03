# Yaga

Yaga is a social help platform with:

- signup/signin
- profile management (`Individual`, `NPO`, `GOV`)
- help request/offer workflows
- help status tracking and counts
- direct user-to-user chat
- appreciation/reward support
- transformation tracking
- prioritized support board
- smart-contract-ready help ledger with transaction audit trail
- MetaMask + ethers lifecycle sync for help actions (optional)

## Project layout

- `backend/yaga`: Rails + MongoDB API
- `frontend`: AngularJS client
- `contracts`: Solidity smart-contract starter (`HelpRegistry`)
- `Docs`: initial design and deployment notes
  - `Docs/smart-contract-v1.md`: chain + API reconciliation flow
  - `Docs/BETA_TESTING_AND_DEPLOYMENT_PLAYBOOK.md`: beta test and deployment runbook
  - `Docs/YAGA_SYSTEM_DESIGN_AND_CURRENT_STATE.md`: detailed architecture and implementation state

## Run locally

1. Start MongoDB on `localhost:27017`.
2. Start backend:
   - `cd backend/yaga`
   - `bundle install`
   - `bundle exec rails s`
3. Serve frontend as static files (for example):
   - `cd frontend`
   - `python3 -m http.server 8080`
   - open `http://localhost:8080`

Frontend expects backend at `http://localhost:3000`.

## Run with Docker

From repo root:

1. `docker compose up --build`
2. Open `http://localhost:8080`

Services:

- Frontend: `http://localhost:8080`
- Backend API: `http://localhost:3000`
- MongoDB: `mongodb://localhost:27017`

## Dependency modernization notes

- Rails upgraded to latest 4.2 patch line (`4.2.11.3`).
- Lockfile regenerated from legacy Windows-only platform to `ruby`.
- Removed legacy doorkeeper/oauth stack (unused by current API implementation).
- Pinned modern-compatible native deps (`nokogiri`, `ffi`) for Ruby 2.6/2.7 era support.
- App configured as Mongoid-first (no ActiveRecord runtime dependency required).
