# Yaga Frontend

AngularJS client for Yaga APIs.

## Features

- Sign up / sign in
- Profile dashboard
- Direct user-to-user chat
- Help request/offer creation
- Help lifecycle actions (take, in progress, complete, cancel)
- Priority-based support listing
- Appreciation/reward submission
- Transformation visibility on user dashboard
- Help-level audit log timeline with blockchain transaction attachments
- MetaMask + `ethers.js` based smart-contract sync for help lifecycle actions

## Configuration

API endpoint is configured in `src/common/common.module.js`:

- `AuthApiPath`: `http://localhost:3000/api/v1`
- `ChainConfig.defaultChainId`: preferred chain ID for wallet sync
- `ChainConfig.contractAddress`: deployed `HelpRegistry` contract address

In the Support page, authenticated users can:

- connect wallet
- save chain + contract config
- enable automatic chain sync for create/take/progress/complete/cancel flows
- review/write ledger events per help

## Run

Serve the folder as static content:

```bash
cd frontend
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

## Docker

From repo root:

```bash
docker compose up --build frontend backend mongodb
```

Open `http://localhost:8080`.
