module Api
  module V1
    class BaseController < ApplicationController
      before_action :authenticate_api_user!

      attr_reader :current_userauth, :current_profile

      private

      def authenticate_api_user!
        token = bearer_token
        @current_userauth = Userauth.where(auth_token: token).first if token.present?

        if @current_userauth.nil? || !@current_userauth.token_active?
          render json: { error: 'Unauthorized' }, status: :unauthorized
          return
        end

        @current_profile = @current_userauth.profile
      end

      # Used by mixed public/authenticated endpoints where auth is optional.
      def authenticate_api_user_if_present!
        token = bearer_token
        return if token.blank?

        userauth = Userauth.where(auth_token: token).first
        return if userauth.nil? || !userauth.token_active?

        @current_userauth = userauth
        @current_profile = userauth.profile
      end

      def bearer_token
        authorization = request.headers['Authorization'].to_s
        return nil unless authorization.start_with?('Bearer ')

        authorization.split(' ', 2).last
      end

      def require_profile!
        return if current_profile.present?

        render json: { error: 'Profile does not exist for current user' }, status: :unprocessable_entity
      end
    end
  end
end
