# Smart Contract v1 Design (Yaga)

This document describes the implemented hybrid architecture:

- on-chain: status transitions + escrow + immutable events
- off-chain: full help content, profile data, chat, media
- bridge: hash pointers + backend ledger reconciliation

## 1) On-chain contract

Contract file: `contracts/HelpRegistry.sol`

Core state:

- `helpId`
- `requester` wallet
- `helper` wallet
- `metadataHash` (hash pointer to off-chain JSON)
- `status`: `Open`, `Matched`, `InProgress`, `Completed`, `Cancelled`
- `rewardEscrowWei`

Core emitted events:

- `HelpCreated`
- `HelpMatched`
- `HelpInProgress`
- `HelpCompleted`
- `HelpCancelled`
- `RewardAdded`

## 2) Backend ledger (implemented)

New Mongoid model:

- `HelpLedgerEvent`

Key fields:

- `help_id`
- `actor_profile_id`
- `action`
- `source_type` (`backend`, `wallet`, `indexer`)
- `chain_state` (`recorded`, `pending`, `submitted`, `confirmed`, `failed`)
- `tx_hash`, `chain_id`, `contract_address`, `block_number`, `log_index`
- `metadata_hash`, `payload_hash`, `onchain_help_id`, `idempotency_key`
- `notes`, `error_message`

Auto-logging now happens for:

- create help
- update help
- take help
- mark in progress
- complete help
- cancel help
- appreciation creation

## 3) API endpoints (implemented)

- `GET /api/v1/helps/:help_id/ledger_events`
- `POST /api/v1/helps/:help_id/ledger_events`
- `PATCH /api/v1/helps/:help_id/ledger_events/:id`

Usage pattern:

1. Backend lifecycle action logs `chain_state=recorded`.
2. Wallet submits tx and records `chain_state=submitted`.
3. Indexer confirms receipt and updates `chain_state=confirmed` with `block_number` and `log_index`.
4. Reconciliation jobs can compare API state vs chain events.

## 4) Frontend support board (implemented)

Support cards now include:

- `View Audit Log`
- timeline of ledger events
- form to attach wallet/indexer transaction details
- MetaMask connect + contract configuration panel
- optional automatic chain sync for lifecycle actions (`create`, `take`, `mark_in_progress`, `complete`, `cancel`)

This enables end-to-end transaction/accountability visibility per help item.

## 5) Recommended next steps

1. Add a dedicated indexer worker that listens to `HelpRegistry` events and updates ledger rows automatically.
2. Add deterministic `idempotency_key` strategy (for example: `chainId:txHash:logIndex`) to avoid duplicate inserts.
3. Add role-based restrictions for manual chain event edits after confirmation.
4. Add nightly reconciliation report:
   - missing on-chain event for backend action
   - missing backend record for on-chain event
   - state drift between help status and latest confirmed chain event
