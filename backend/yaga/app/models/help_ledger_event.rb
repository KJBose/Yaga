class HelpLedgerEvent
  include Mongoid::Document
  include Mongoid::Timestamps

  ACTIONS = %w[
    create
    update
    take
    mark_in_progress
    complete
    cancel
    appreciation
    anchor
    reconcile
  ].freeze
  CHAIN_STATES = %w[recorded pending submitted confirmed failed].freeze
  SOURCE_TYPES = %w[backend wallet indexer].freeze

  field :action, type: String
  field :source_type, type: String, default: 'backend'
  field :chain_state, type: String, default: 'recorded'
  field :tx_hash, type: String
  field :chain_id, type: Integer
  field :contract_address, type: String
  field :block_number, type: Integer
  field :log_index, type: Integer
  field :wallet_address, type: String
  field :metadata_hash, type: String
  field :payload_hash, type: String
  field :onchain_help_id, type: Integer
  field :idempotency_key, type: String
  field :notes, type: String
  field :error_message, type: String

  belongs_to :help, class_name: 'Help', inverse_of: :ledger_events
  belongs_to :actor_profile, class_name: 'Profile', inverse_of: :help_ledger_events

  validates :action, presence: true, inclusion: { in: ACTIONS }
  validates :source_type, presence: true, inclusion: { in: SOURCE_TYPES }
  validates :chain_state, presence: true, inclusion: { in: CHAIN_STATES }
  validates :idempotency_key, uniqueness: { scope: :help_id }, allow_blank: true

  index({ help_id: 1, created_at: -1 }, background: true)
  index({ actor_profile_id: 1, created_at: -1 }, background: true)
  index({ tx_hash: 1 }, background: true, sparse: true)
  index({ help_id: 1, idempotency_key: 1 }, background: true, sparse: true)

  before_validation :normalize_values
  after_save :sync_help_ledger_snapshot

  def as_api_json
    {
      id: id.to_s,
      help_id: help_id.to_s,
      actor_profile_id: actor_profile_id&.to_s,
      action: action,
      source_type: source_type,
      chain_state: chain_state,
      tx_hash: tx_hash,
      chain_id: chain_id,
      contract_address: contract_address,
      block_number: block_number,
      log_index: log_index,
      wallet_address: wallet_address,
      metadata_hash: metadata_hash,
      payload_hash: payload_hash,
      onchain_help_id: onchain_help_id,
      idempotency_key: idempotency_key,
      notes: notes,
      error_message: error_message,
      created_at: created_at,
      updated_at: updated_at
    }
  end

  private

  def normalize_values
    self.action = action.to_s.strip
    self.source_type = source_type.to_s.strip.presence || 'backend'
    self.chain_state = chain_state.to_s.strip.presence || 'recorded'
    self.tx_hash = tx_hash.to_s.strip.presence
    self.contract_address = contract_address.to_s.strip.presence
    self.wallet_address = wallet_address.to_s.strip.presence
    self.metadata_hash = metadata_hash.to_s.strip.presence
    self.payload_hash = payload_hash.to_s.strip.presence
    self.idempotency_key = idempotency_key.to_s.strip.presence
    self.notes = notes.to_s.strip.presence
    self.error_message = error_message.to_s.strip.presence
  end

  def sync_help_ledger_snapshot
    return if help.nil?

    updates = {
      ledger_event_count: help.ledger_events.count,
      last_chain_action: action,
      last_chain_state: chain_state,
      last_chain_event_at: updated_at
    }
    updates[:last_chain_tx_hash] = tx_hash if tx_hash.present?
    updates[:chain_contract_address] = contract_address if contract_address.present?
    updates[:chain_id] = chain_id if chain_id.present?
    updates[:onchain_help_id] = onchain_help_id if onchain_help_id.present?
    updates[:metadata_hash] = metadata_hash if metadata_hash.present?

    help.set(updates)
  end
end
