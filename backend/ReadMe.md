# Yaga Backend

Ruby on Rails + MongoDB backend for Yaga.

## Implemented objectives

- Multi-user signup/signin with token auth.
- Profile creation and update for `Individual`, `NPO`, and `GOV`.
- Help lifecycle support:
  - Create help request/offer
  - Take help
  - Track status (`open`, `matched`, `in_progress`, `completed`, `cancelled`)
- Help counters (`requested`, `offered`, `taken`, `given`) per user profile.
- Appreciations/rewards after completed help.
- Transformation view as completed helps where the user led or contributed.
- Priority scoring of help activities based on:
  - urgency
  - scale
  - required resources
  - time available
  - supporter count

## API base

`http://localhost:3000/api/v1`

## Auth endpoints

- `POST /auth/signup`
- `POST /auth/signin`
- `DELETE /auth/signout`
- `GET /auth/me`

Use `Authorization: Bearer <token>` for protected endpoints.

## Profile endpoints

- `GET /profile`
- `PATCH /profile`

## Help endpoints

- `GET /helps`
- `GET /helps/prioritized`
- `GET /helps/my`
- `GET /helps/:id`
- `POST /helps`
- `PATCH /helps/:id`
- `POST /helps/:id/take`
- `POST /helps/:id/mark_in_progress`
- `POST /helps/:id/complete`
- `POST /helps/:id/cancel`
- `GET /helps/:id/ledger_events` (public audit trail view)
- `POST /helps/:id/ledger_events` (participant logs wallet/indexer tx data)
- `PATCH /helps/:help_id/ledger_events/:id` (participant updates chain confirmation state)

## Appreciation / transformation endpoints

- `GET /appreciations`
- `POST /appreciations`
- `GET /transformations`

## Smart-contract audit model

- Every backend help lifecycle transition is auto-logged as a ledger event.
- Ledger supports chain reconciliation fields:
  - `tx_hash`
  - `chain_id`
  - `contract_address`
  - `onchain_help_id`
  - `metadata_hash`
  - `chain_state` (`recorded`, `pending`, `submitted`, `confirmed`, `failed`)
- Solidity starter contract is in `../contracts/HelpRegistry.sol`.

## Runtime notes

- Uses MongoDB via Mongoid (`MONGODB_URI` env var supported in all environments).
- ActiveRecord/sqlite runtime dependency is removed.
- Doorkeeper/oauth legacy stack is removed; token auth is provided by `/api/v1/auth/*`.

## Docker backend run

From repo root:

```bash
docker compose up --build backend mongodb
```
