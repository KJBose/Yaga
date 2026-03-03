module Api
  module V1
    class AuthController < BaseController
      skip_before_action :authenticate_api_user!, only: [:signup, :signin]

      def signup
        userauth = Userauth.new(
          email: auth_params[:email].to_s.strip.downcase,
          password: auth_params[:password],
          password_confirmation: auth_params[:password_confirmation]
        )

        profile = userauth.build_profile(profile_params)

        if userauth.valid? && profile.valid?
          userauth.save!
          profile.save! unless profile.persisted?
          token = userauth.generate_auth_token!

          render json: success_payload(userauth, token), status: :created
          return
        end

        render json: { errors: userauth.errors.full_messages + profile.errors.full_messages }, status: :unprocessable_entity
      end

      def signin
        email = auth_params[:email].to_s.strip.downcase
        password = auth_params[:password].to_s
        userauth = Userauth.where(email: email).first

        if userauth.nil? || !userauth.valid_password?(password)
          render json: { error: 'Invalid email/password combination' }, status: :unauthorized
          return
        end

        token = userauth.generate_auth_token!
        render json: success_payload(userauth, token)
      end

      def signout
        current_userauth.clear_auth_token!
        head :no_content
      end

      def me
        render json: {
          user: {
            id: current_userauth.id.to_s,
            email: current_userauth.email
          },
          profile: current_profile&.as_api_json,
          help_counts: current_profile&.help_counts || {},
          appreciation: current_profile&.appreciation_summary || {}
        }
      end

      private

      def success_payload(userauth, token)
        {
          token: token,
          user: {
            id: userauth.id.to_s,
            email: userauth.email
          },
          profile: userauth.profile&.as_api_json,
          help_counts: userauth.profile&.help_counts || {},
          appreciation: userauth.profile&.appreciation_summary || {}
        }
      end

      def auth_params
        root = params[:auth].is_a?(ActionController::Parameters) ? params[:auth] : params
        root.permit(:email, :password, :password_confirmation)
      end

      def profile_params
        root = params[:auth].is_a?(ActionController::Parameters) ? params[:auth] : params
        root.permit(
          :first_name,
          :last_name,
          :phone,
          :user_type,
          :organization_name,
          :bio,
          :profile_visibility,
          :network_visibility
        )
      end
    end
  end
end
