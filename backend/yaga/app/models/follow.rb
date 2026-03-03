class Follow
  include Mongoid::Document
  include Mongoid::Timestamps

  belongs_to :follower, class_name: 'Profile', inverse_of: :following_relationships, foreign_key: 'follower_profile_id'
  belongs_to :followed, class_name: 'Profile', inverse_of: :follower_relationships, foreign_key: 'followed_profile_id'

  validates :follower, :followed, presence: true
  validate :follower_and_followed_must_differ
  validate :relationship_must_be_unique

  index({ follower_profile_id: 1, followed_profile_id: 1 }, unique: true, background: true)
  index({ follower_profile_id: 1, created_at: -1 }, background: true)
  index({ followed_profile_id: 1, created_at: -1 }, background: true)

  private

  def follower_and_followed_must_differ
    return if follower_profile_id != followed_profile_id

    errors.add(:followed, 'must be a different user')
  end

  def relationship_must_be_unique
    return if follower_profile_id.blank? || followed_profile_id.blank?

    scope = self.class.where(follower_profile_id: follower_profile_id, followed_profile_id: followed_profile_id)
    scope = scope.where(:id.ne => id) if persisted?
    return unless scope.exists?

    errors.add(:base, 'Already following this user')
  end
end
