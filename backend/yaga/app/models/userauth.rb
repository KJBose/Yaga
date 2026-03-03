require 'securerandom'

class Userauth
  include Mongoid::Document
  include Mongoid::Timestamps
  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable and :omniauthable
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :trackable, :validatable

  TOKEN_TTL = 30.days

  has_one :profile, class_name: 'Profile', inverse_of: :userauth, dependent: :destroy

  ## Database authenticatable
  field :email,              type: String, default: ""
  field :encrypted_password, type: String, default: ""
  field :auth_token, type: String
  field :auth_token_generated_at, type: Time

  ## Recoverable
  field :reset_password_token,   type: String
  field :reset_password_sent_at, type: Time

  ## Rememberable
  field :remember_created_at, type: Time

  ## Trackable
  field :sign_in_count,      type: Integer, default: 0
  field :current_sign_in_at, type: Time
  field :last_sign_in_at,    type: Time
  field :current_sign_in_ip, type: String
  field :last_sign_in_ip,    type: String

  ## Confirmable
  # field :confirmation_token,   type: String
  # field :confirmed_at,         type: Time
  # field :confirmation_sent_at, type: Time
  # field :unconfirmed_email,    type: String # Only if using reconfirmable

  ## Lockable
  # field :failed_attempts, type: Integer, default: 0 # Only if lock strategy is :failed_attempts
  # field :unlock_token,    type: String # Only if unlock strategy is :email or :both
  # field :locked_at,       type: Time

  index({ email: 1 }, unique: true, background: true)
  index({ auth_token: 1 }, sparse: true, background: true)

  def generate_auth_token!
    update_attributes!(
      auth_token: SecureRandom.hex(32),
      auth_token_generated_at: Time.current
    )
    auth_token
  end

  def clear_auth_token!
    update_attributes!(auth_token: nil, auth_token_generated_at: nil)
  end

  def token_active?
    return false if auth_token.blank? || auth_token_generated_at.blank?

    auth_token_generated_at >= TOKEN_TTL.ago
  end
end
