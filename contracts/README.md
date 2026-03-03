# Yaga Smart Contract v1

This folder contains a starter smart-contract layer for auditable help workflows.

- Contract: `HelpRegistry.sol`
- Solidity: `^0.8.20`
- On-chain data is intentionally minimal:
  - requester/helper wallet
  - metadata hash
  - status transitions
  - escrow/reward value

Do not store personal data or chat content on-chain.

## Event model

`HelpRegistry` emits:

- `HelpCreated`
- `HelpMatched`
- `HelpInProgress`
- `HelpCompleted`
- `HelpCancelled`
- `RewardAdded`

These events are the source of truth for chain accounting.

## Backend reconciliation mapping

Yaga backend now supports a help-scoped ledger API:

- `GET /api/v1/helps/:help_id/ledger_events`
- `POST /api/v1/helps/:help_id/ledger_events`
- `PATCH /api/v1/helps/:help_id/ledger_events/:id`

Suggested integration sequence:

1. User performs action in UI (`create`, `take`, `complete`, etc.).
2. Backend records lifecycle event automatically (`source_type=backend`, `chain_state=recorded`).
3. Wallet/indexer submits transaction hash to ledger endpoint:
   - `source_type=wallet` or `source_type=indexer`
   - `chain_state=submitted|pending|confirmed|failed`
   - `tx_hash`, `chain_id`, `contract_address`, `onchain_help_id`, `metadata_hash`
4. Indexer confirms block/log and updates same ledger row to `confirmed`.

This gives an auditable, append-only timeline that reconciles:

- API actions
- blockchain transaction hashes
- final confirmed chain events

## Frontend wallet sync

Support page (`/#/support`) now includes a **Smart Contract Sync** panel for authenticated users.

After deploying `HelpRegistry`:

1. connect MetaMask in browser
2. enter `contract address` and `chain ID`
3. enable automatic chain sync toggle
4. create/take/progress/complete/cancel actions will:
   - execute backend lifecycle
   - attempt on-chain transaction
   - store tx metadata in backend ledger

## Metadata hash guidance

Use a deterministic hash for off-chain JSON metadata:

- Example metadata fields:
  - `title`
  - `description`
  - `category`
  - `urgency/scale/resources/time`
  - `media_url`
- Serialize consistently and hash (for example, keccak256 of canonical JSON).
- Store raw metadata in object store/IPFS; store hash on-chain and in ledger logs.
