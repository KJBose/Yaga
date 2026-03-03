class Appreciation
  include Mongoid::Document
  include Mongoid::Timestamps

  field :note, type: String
  field :reward_points, type: Integer, default: 1

  belongs_to :help, class_name: 'Help', inverse_of: :appreciations
  belongs_to :from_profile, class_name: 'Profile', inverse_of: :given_appreciations
  belongs_to :to_profile, class_name: 'Profile', inverse_of: :received_appreciations

  validates :reward_points, numericality: { greater_than_or_equal_to: 1, less_than_or_equal_to: 100 }
  validates :note, length: { maximum: 600 }

  index({ help_id: 1, created_at: -1 }, background: true)
  index({ to_profile_id: 1, created_at: -1 }, background: true)
  index({ from_profile_id: 1, created_at: -1 }, background: true)

  def as_api_json
    {
      id: id.to_s,
      note: note,
      reward_points: reward_points,
      help_id: help_id.to_s,
      from_profile: from_profile&.as_api_json,
      to_profile: to_profile&.as_api_json,
      created_at: created_at
    }
  end
end
