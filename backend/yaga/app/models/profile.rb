class Profile
  include Mongoid::Document
  include Mongoid::Timestamps

  USER_TYPES = %w[Individual NPO GOV].freeze
  VISIBILITY_LEVELS = %w[public members_only private].freeze

  field :first_name, type: String
  field :last_name, type: String
  field :phone, type: String
  field :user_type, type: String, default: 'Individual'
  field :organization_name, type: String
  field :bio, type: String
  field :profile_visibility, type: String, default: 'public'
  field :network_visibility, type: String, default: 'public'

  belongs_to :userauth, class_name: 'Userauth', inverse_of: :profile

  has_many :requested_helps, class_name: 'Help', inverse_of: :requester, dependent: :nullify
  has_many :offered_helps, class_name: 'Help', inverse_of: :helper, dependent: :nullify
  has_many :help_ledger_events, class_name: 'HelpLedgerEvent', inverse_of: :actor_profile, foreign_key: 'actor_profile_id', dependent: :nullify
  has_many :given_appreciations, class_name: 'Appreciation', inverse_of: :from_profile, dependent: :nullify
  has_many :received_appreciations, class_name: 'Appreciation', inverse_of: :to_profile, dependent: :nullify
  has_many :sent_chat_messages, class_name: 'ChatMessage', inverse_of: :sender, foreign_key: 'sender_profile_id', dependent: :nullify
  has_many :following_relationships, class_name: 'Follow', inverse_of: :follower, foreign_key: 'follower_profile_id', dependent: :destroy
  has_many :follower_relationships, class_name: 'Follow', inverse_of: :followed, foreign_key: 'followed_profile_id', dependent: :destroy

  validates :first_name, :last_name, presence: true
  validates :user_type, inclusion: { in: USER_TYPES }
  validates :profile_visibility, inclusion: { in: VISIBILITY_LEVELS }
  validates :network_visibility, inclusion: { in: VISIBILITY_LEVELS }

  before_validation :apply_visibility_defaults

  index({ userauth_id: 1 }, unique: true, background: true)
  index({ user_type: 1 }, background: true)

  def full_name
    [first_name, last_name].compact.join(' ').strip
  end

  def help_counts
    {
      requested: Help.where(requester_id: id, help_type: 'request').count,
      offered: Help.where(requester_id: id, help_type: 'offer').count,
      taken: Help.where(helper_id: id, :status.in => %w[matched in_progress completed]).count,
      given: Help.where(helper_id: id, status: 'completed').count
    }
  end

  def appreciation_summary
    received = Appreciation.where(to_profile_id: id)
    given = Appreciation.where(from_profile_id: id)

    {
      received_count: received.count,
      received_points: received.sum(:reward_points),
      given_count: given.count,
      given_points: given.sum(:reward_points)
    }
  end

  def followers_count
    Follow.where(followed_profile_id: id).count
  end

  def following_count
    Follow.where(follower_profile_id: id).count
  end

  def follows?(other_profile)
    return false if other_profile.nil?

    Follow.where(follower_profile_id: id, followed_profile_id: other_profile.id).exists?
  end

  def followed_by?(other_profile)
    return false if other_profile.nil?

    Follow.where(follower_profile_id: other_profile.id, followed_profile_id: id).exists?
  end

  def profile_visible_to?(viewer_profile)
    return true if profile_visibility == 'public'
    return viewer_profile.present? if profile_visibility == 'members_only'
    return false if viewer_profile.nil?
    return true if viewer_profile.id == id

    followed_by?(viewer_profile)
  end

  def network_visible_to?(viewer_profile)
    return true if network_visibility == 'public'
    return viewer_profile.present? if network_visibility == 'members_only'
    return false if viewer_profile.nil?
    return true if viewer_profile.id == id

    followed_by?(viewer_profile)
  end

  def as_api_json
    {
      id: id.to_s,
      first_name: first_name,
      last_name: last_name,
      full_name: full_name,
      phone: phone,
      user_type: user_type,
      organization_name: organization_name,
      bio: bio,
      profile_visibility: profile_visibility,
      network_visibility: network_visibility,
      follower_count: followers_count,
      following_count: following_count
    }
  end

  private

  def apply_visibility_defaults
    self.profile_visibility = 'public' if profile_visibility.blank?
    self.network_visibility = 'public' if network_visibility.blank?
  end
end
