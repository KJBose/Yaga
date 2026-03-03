class ChatMessage
  include Mongoid::Document
  include Mongoid::Timestamps

  field :body, type: String
  field :sender_profile_id, type: String

  belongs_to :chat_conversation, class_name: 'ChatConversation', inverse_of: :chat_messages
  belongs_to :sender, class_name: 'Profile', inverse_of: :sent_chat_messages, foreign_key: 'sender_profile_id'

  validates :body, presence: true, length: { maximum: 2000 }
  validates :sender_profile_id, presence: true
  validate :sender_is_participant

  index({ chat_conversation_id: 1, created_at: 1 }, background: true)
  index({ sender_profile_id: 1, created_at: -1 }, background: true)

  before_validation :normalize_body

  def as_api_json(current_profile = nil)
    {
      id: id.to_s,
      chat_id: chat_conversation_id.to_s,
      body: body,
      sender_id: sender_profile_id,
      sender: sender&.as_api_json,
      mine: current_profile.present? && current_profile.id.to_s == sender_profile_id,
      created_at: created_at,
      updated_at: updated_at
    }
  end

  private

  def normalize_body
    self.body = body.to_s.strip
  end

  def sender_is_participant
    return if chat_conversation.nil? || sender_profile_id.blank?
    return if chat_conversation.include_profile?(sender_profile_id)

    errors.add(:sender_profile_id, 'must belong to this conversation')
  end
end
