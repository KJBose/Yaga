class ChatConversation
  include Mongoid::Document
  include Mongoid::Timestamps

  field :conversation_key, type: String
  field :participant_ids, type: Array, default: []
  field :last_message_at, type: Time
  field :last_message_preview, type: String

  has_many :chat_messages, class_name: 'ChatMessage', inverse_of: :chat_conversation, dependent: :destroy

  validates :conversation_key, presence: true, uniqueness: true
  validates :participant_ids, presence: true
  validate :participant_ids_are_valid

  index({ conversation_key: 1 }, unique: true, background: true)
  index({ participant_ids: 1, last_message_at: -1 }, background: true)
  index({ last_message_at: -1 }, background: true)

  before_validation :normalize_participants
  before_create :set_default_last_message_at

  def self.find_or_create_direct!(profile_id, other_profile_id)
    ids = normalized_pair(profile_id, other_profile_id)
    key = ids.join(':')
    existing = where(conversation_key: key).first
    return existing if existing.present?

    create!(conversation_key: key, participant_ids: ids)
  rescue Mongo::Error::OperationFailure => e
    raise unless e.message.to_s.include?('E11000')

    where(conversation_key: key).first
  end

  def self.normalized_pair(*profile_ids)
    profile_ids.flatten.map(&:to_s).reject(&:blank?).uniq.sort
  end

  def include_profile?(profile_id)
    participant_ids.include?(profile_id.to_s)
  end

  def participant_profiles
    Profile.where(:id.in => participant_ids).to_a
  end

  def other_participant_for(profile)
    current_id = profile&.id.to_s
    participant_profiles.find { |participant| participant.id.to_s != current_id }
  end

  def touch_last_message!(message_body)
    update_attributes!(
      last_message_at: Time.current,
      last_message_preview: message_body.to_s.strip[0, 160]
    )
  end

  def as_api_json_for(current_profile)
    participants = participant_profiles.map(&:as_api_json)
    other_participant = participants.find { |participant| participant[:id] != current_profile.id.to_s }

    {
      id: id.to_s,
      conversation_key: conversation_key,
      participant_ids: participant_ids,
      participants: participants,
      other_participant: other_participant,
      last_message_at: last_message_at || updated_at,
      last_message_preview: last_message_preview,
      created_at: created_at,
      updated_at: updated_at
    }
  end

  private

  def normalize_participants
    self.participant_ids = self.class.normalized_pair(participant_ids || [])
    self.conversation_key = participant_ids.join(':') if participant_ids.size == 2
  end

  def set_default_last_message_at
    self.last_message_at ||= Time.current
  end

  def participant_ids_are_valid
    if participant_ids.size != 2
      errors.add(:participant_ids, 'must include exactly two unique users')
      return
    end

    existing_count = Profile.where(:id.in => participant_ids).count
    return if existing_count == 2

    errors.add(:participant_ids, 'must reference valid user profiles')
  end
end
