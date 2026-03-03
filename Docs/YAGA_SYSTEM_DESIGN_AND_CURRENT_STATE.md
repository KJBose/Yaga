# Yaga System Design and Current State

## 1. Document Intent

This document is the canonical AI-readable technical reference for Yaga.

It explains:

1. product objectives
2. architecture and code structure
3. data and API contracts
4. current implementation state (what exists now)
5. known gaps and next-step roadmap

Repository base: `https://github.com/KJBose/Yaga.git`

Workspace used for this state: `/Users/jkarnati/workspaces/yaga-dedicated`

---

## 2. Product Objectives

Yaga is a social-impact network where users can:

- request help
- offer help
- match and complete help actions
- appreciate contributors
- follow users and track impact
- communicate through chat

Core objective:

- make every help interaction visible, accountable, and progressively auditable
- support optional blockchain anchoring for strong transaction traceability

---

## 3. High-Level Architecture

Yaga is a three-layer system:

1. Frontend: AngularJS SPA served by Nginx
2. Backend: Rails 4.2 API with Mongoid/MongoDB
3. Optional chain layer: Solidity contract (`HelpRegistry`) + wallet integration

### Runtime ports

- frontend: `8080`
- backend: `3000`
- MongoDB: `27017`

### Container orchestration

`docker-compose.yml` defines services:

- `frontend` (Nginx static host)
- `backend` (Rails API)
- `mongodb` (`mongo:6`)

---

## 4. Repository Structure

- `backend/yaga`: Rails app
- `frontend`: AngularJS SPA
- `contracts`: Solidity contract starter + contract docs
- `Docs`: project docs and runbooks

Important entry files:

- backend routes: `backend/yaga/config/routes.rb`
- backend base auth controller: `backend/yaga/app/controllers/api/v1/base_controller.rb`
- frontend app shell: `frontend/index.html`
- frontend state routes: `frontend/src/public/public.routes.js`

---

## 5. Backend Design

## 5.1 API routing

Namespace: `/api/v1`

Primary route groups:

- auth: signup/signin/signout/me
- profile: own profile + public profile
- helps: lifecycle actions + prioritization + user scope
- help ledger events: audit and chain reconciliation records
- chats: conversations and message operations
- social: follow graph operations
- appreciations: reward and recognition actions
- transformations: completed impact stories
- leaderboard: ranking view

CORS preflight OPTIONS wildcard route exists.

---

## 5.2 Auth model

Authentication is token-based:

- Bearer token issued at signin/signup
- token validated in `Api::V1::BaseController`
- protected routes require active token and profile context

`authenticate_api_user_if_present!` enables mixed public/auth flows.

---

## 5.3 Domain models (current)

### Userauth

- identity credentials, token state, auth methods
- has one `Profile`

### Profile

- person/org identity fields
- visibility settings (`public`, `members_only`, `private`)
- associations: helps, follows, chat, appreciations

### Help

Core help fields:

- title/description/category
- type: `request` or `offer`
- status: `open`, `matched`, `in_progress`, `completed`, `cancelled`
- prioritization factors: urgency, scale, required resources, time, supporters
- anonymous requester support

Chain-aware fields:

- `ledger_event_count`
- `last_chain_action`, `last_chain_state`, `last_chain_tx_hash`, `last_chain_event_at`
- `chain_contract_address`, `chain_id`, `onchain_help_id`, `metadata_hash`

### HelpLedgerEvent

Append-only audit and reconciliation model with:

- action (`create`, `take`, `complete`, etc.)
- source (`backend`, `wallet`, `indexer`)
- chain state (`recorded`, `submitted`, `confirmed`, etc.)
- tx metadata (`tx_hash`, `chain_id`, `block_number`, `log_index`, etc.)
- optional idempotency key and notes

### Follow

- follower/followed profile relationship

### ChatConversation / ChatMessage

- direct user communication state and message content

### Appreciation

- post-completion recognition and reward points

Legacy models (`user.rb`, `yagauser.rb`) exist but primary social platform uses `Userauth` + `Profile`.

---

## 5.4 Help lifecycle behavior

Current authoritative lifecycle is backend-managed.

Transitions:

1. `create`
2. optional `take` by helper
3. `mark_in_progress`
4. `complete` or `cancel`

Authorization rules:

- requester cannot take own help
- non-participants cannot mutate participant-only actions
- only participant can log/update ledger events

Every backend lifecycle mutation writes a corresponding `HelpLedgerEvent` (`source=backend`, `chain_state=recorded`).

---

## 5.5 Ledger and chain reconciliation

Endpoints:

- `GET /api/v1/helps/:help_id/ledger_events`
- `POST /api/v1/helps/:help_id/ledger_events`
- `PATCH /api/v1/helps/:help_id/ledger_events/:id`

Pattern:

1. backend logs system event
2. wallet/indexer appends tx record (`submitted`)
3. later update to `confirmed` with block/log index

This supports auditability without forcing immediate on-chain dependency.

---

## 6. Frontend Design

## 6.1 Framework and modules

- AngularJS 1.5 + UI-Router
- modules: `yaga`, `public`, `common`

Main states:

- `public.index` (`/`)
- `public.home` (`/home`, authenticated)
- `public.menu` (`/support`)
- `public.profile` (`/profile`, authenticated)
- `public.userProfile` (`/users/:profileId`)
- `public.chat` (`/chat`)
- auth screens (`/signin`, `/create-account`)

---

## 6.2 Services

### AuthService

- signup/signin/signout/me
- token storage in `localStorage`

### HelpService

- help CRUD/lifecycle APIs
- appreciation create
- help ledger event APIs

### Web3HelpService

- MetaMask connect flow
- preferred chain switching
- `HelpRegistry` contract calls via ethers
- metadata hash generation
- tx result extraction for ledger writing

### SocialService / UserDataService / ChatService

- social graph, dashboard, profile retrieval, chat workflows

---

## 6.3 Support page behavior (current)

Support page contains:

1. support hero and composer
2. help posting form
3. prioritized help cards
4. audit log viewer per help
5. manual wallet/indexer event logging form
6. Smart Contract Sync panel (authenticated users)

Smart Contract Sync panel allows:

- contract address + chain ID config
- wallet connect/switch chain
- enable auto chain sync for lifecycle actions

When enabled:

- backend action executes first
- frontend attempts matching on-chain action
- tx metadata posted to backend ledger

If chain action fails:

- backend flow remains successful
- warning shown in UI

---

## 7. Smart Contract Layer

Contract file:

- `contracts/HelpRegistry.sol`

Current contract supports:

- `createHelp(metadataHash)`
- `offerHelp(helpId)`
- `markInProgress(helpId)`
- `completeHelp(helpId)`
- `cancelHelp(helpId)`
- `addReward(helpId)`

Events emitted for each state transition and reward funding.

Design intent:

- minimal on-chain data
- keep PII and rich content off-chain
- use metadata hash pointers and backend ledger reconciliation

---

## 8. Current State Summary (as of this code)

### Implemented and usable

1. end-to-end auth and profile flows
2. support board with help lifecycle transitions
3. chat and social graph features
4. leaderboard and transformations
5. immutable-style help ledger model and APIs
6. optional wallet-connected chain sync in UI
7. Dockerized local runtime for full stack

### Implemented as starter/foundation

1. Solidity contract exists but deployment tooling is minimal
2. chain indexing worker is not yet implemented
3. automated reconciliation jobs are not yet implemented

### Known operational caveats

1. production security hardening is pending (CORS tightening, rate limits, etc.)
2. legacy files coexist with new social platform code
3. automated test coverage should be expanded (API + E2E)

---

## 9. AI Agent Onboarding Guide

For any AI agent extending this repo:

1. start with routes + primary models
2. treat `Userauth + Profile + Help + HelpLedgerEvent` as core domain
3. preserve backend lifecycle as source-of-truth
4. keep chain integration additive and failure-tolerant
5. maintain cache-busting version updates in `frontend/index.html` and route template URLs

Recommended reading order:

1. `backend/yaga/config/routes.rb`
2. `backend/yaga/app/controllers/api/v1/base_controller.rb`
3. `backend/yaga/app/controllers/api/v1/helps_controller.rb`
4. `backend/yaga/app/models/help.rb`
5. `backend/yaga/app/models/help_ledger_event.rb`
6. `frontend/src/public/public.routes.js`
7. `frontend/src/public/menu/menu.controller.js`
8. `frontend/src/common/help.service.js`
9. `frontend/src/common/web3-help.service.js`
10. `contracts/HelpRegistry.sol`

---

## 10. Future Roadmap (recommended)

## 10.1 Product hardening

1. strict validation and security review
2. robust API regression and UI E2E tests
3. load and reliability benchmarking

## 10.2 Chain maturity

1. deploy managed indexer listener for contract events
2. deterministic idempotency policy enforcement
3. reconciliation dashboards and alerting

## 10.3 Platform evolution

1. notifications and activity digest
2. richer recommendation ranking for support feed
3. moderation/admin controls and abuse handling

---

## 11. Definition of Done for "Production-Ready"

1. automated CI/CD with rollback
2. staging parity with production
3. full observability and alerting
4. security baseline completed
5. data backup/restore tested
6. support runbook and on-call processes in place

