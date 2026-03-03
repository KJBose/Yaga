class Help
  include Mongoid::Document
  include Mongoid::Timestamps

  HELP_TYPES = %w[request offer].freeze
  STATUSES = %w[open matched in_progress completed cancelled].freeze

  field :title, type: String
  field :description, type: String
  field :category, type: String
  field :help_type, type: String, default: 'request'
  field :status, type: String, default: 'open'
  field :urgency, type: Integer, default: 1
  field :scale, type: Integer, default: 1
  field :required_resources, type: Integer, default: 1
  field :time_available_hours, type: Integer, default: 24
  field :supporters, type: Integer, default: 0
  field :media_url, type: String
  field :acknowledgement_url, type: String
  field :anonymous_requester, type: Mongoid::Boolean, default: false
  field :requester_name, type: String
  field :requester_contact, type: String
  field :priority_score, type: Float, default: 0.0
  field :completed_at, type: Time
  field :ledger_event_count, type: Integer, default: 0
  field :last_chain_action, type: String
  field :last_chain_state, type: String
  field :last_chain_tx_hash, type: String
  field :last_chain_event_at, type: Time
  field :chain_contract_address, type: String
  field :chain_id, type: Integer
  field :onchain_help_id, type: Integer
  field :metadata_hash, type: String

  belongs_to :requester, class_name: 'Profile', inverse_of: :requested_helps
  belongs_to :helper, class_name: 'Profile', inverse_of: :offered_helps

  has_many :appreciations, class_name: 'Appreciation', inverse_of: :help, dependent: :destroy
  has_many :ledger_events, class_name: 'HelpLedgerEvent', inverse_of: :help, dependent: :destroy

  validates :title, :description, :help_type, :status, presence: true
  validates :help_type, inclusion: { in: HELP_TYPES }
  validates :status, inclusion: { in: STATUSES }
  validates :urgency, :scale, :required_resources, inclusion: { in: 1..5 }
  validates :time_available_hours, numericality: { greater_than_or_equal_to: 1 }
  validates :supporters, numericality: { greater_than_or_equal_to: 0 }
  validate :requester_presence_for_non_anonymous

  index({ requester_id: 1, created_at: -1 }, background: true)
  index({ helper_id: 1, created_at: -1 }, background: true)
  index({ status: 1, priority_score: -1 }, background: true)
  index({ category: 1 }, background: true)

  before_validation :normalize_values
  before_save :compute_priority_score

  def open?
    status == 'open'
  end

  def completed?
    status == 'completed'
  end

  def involve_profile?(profile)
    return false if profile.nil?

    requester_id == profile.id || helper_id == profile.id
  end

  def transition_to!(new_status)
    self.status = new_status
    self.completed_at = Time.current if new_status == 'completed'
    save!
  end

  def as_api_json
    {
      id: id.to_s,
      title: title,
      description: description,
      category: category,
      help_type: help_type,
      status: status,
      urgency: urgency,
      scale: scale,
      required_resources: required_resources,
      time_available_hours: time_available_hours,
      supporters: supporters,
      priority_score: priority_score,
      media_url: media_url,
      acknowledgement_url: acknowledgement_url,
      anonymous_requester: anonymous_requester,
      requester_name: requester_name,
      requester_contact: requester_contact,
      requester_label: requester_label,
      completed_at: completed_at,
      requester: anonymous_requester ? nil : requester&.as_api_json,
      helper: helper&.as_api_json,
      ledger: ledger_summary,
      chain: chain_summary,
      created_at: created_at,
      updated_at: updated_at
    }
  end

  def ledger_summary
    {
      event_count: ledger_event_count.to_i,
      last_action: last_chain_action,
      last_state: last_chain_state,
      last_tx_hash: last_chain_tx_hash,
      last_event_at: last_chain_event_at
    }
  end

  def chain_summary
    {
      chain_id: chain_id,
      contract_address: chain_contract_address,
      onchain_help_id: onchain_help_id,
      metadata_hash: metadata_hash
    }
  end

  private

  def normalize_values
    self.urgency = normalize_rank(urgency)
    self.scale = normalize_rank(scale)
    self.required_resources = normalize_rank(required_resources)
    self.time_available_hours = [time_available_hours.to_i, 1].max
    self.supporters = [supporters.to_i, 0].max
    self.requester_name = requester_name.to_s.strip
    self.requester_contact = requester_contact.to_s.strip
    self.anonymous_requester = !!anonymous_requester
    self.requester_name = nil if requester_name.blank?
    self.requester_contact = nil if requester_contact.blank?
  end

  def requester_label
    if anonymous_requester
      return requester_name if requester_name.present?
      return 'Anonymous Guest'
    end

    return requester.full_name if requester.present?
    return requester_name if requester_name.present?

    'Community member'
  end

  def normalize_rank(value)
    value = value.to_i
    return 1 if value < 1
    return 5 if value > 5

    value
  end

  # Higher score means higher priority in support listings.
  def compute_priority_score
    time_pressure_raw = [72 - time_available_hours, 0].max
    time_pressure = [time_pressure_raw, 72].min.to_f / 72.0 * 5.0
    supporter_score = [supporters, 25].min.to_f / 25.0 * 5.0

    self.priority_score = (
      (urgency * 0.35) +
      (scale * 0.2) +
      (required_resources * 0.15) +
      (time_pressure * 0.2) +
      (supporter_score * 0.1)
    ).round(2)
  end

  def requester_presence_for_non_anonymous
    return if requester.present?
    return if anonymous_requester

    errors.add(:requester, 'must exist unless this help is anonymous')
  end
end
