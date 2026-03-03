module Api
  module V1
    class ProfilesController < BaseController
      skip_before_action :authenticate_api_user!, only: [:public_show]
      before_action :authenticate_api_user_if_present!, only: [:public_show]

      def show
        profile = current_profile || current_userauth.build_profile
        render json: profile_payload(profile)
      end

      def public_show
        profile = Profile.where(id: params[:id]).first

        if profile.nil?
          render json: { error: 'Profile not found' }, status: :not_found
          return
        end

        unless profile.profile_visible_to?(current_profile)
          render json: {
            error: 'This profile is private based on user settings',
            profile_visibility: profile.profile_visibility
          }, status: :forbidden
          return
        end

        render json: profile_payload(profile)
      end

      def update
        profile = current_profile || current_userauth.build_profile

        if profile.update_attributes(profile_params)
          render json: profile_payload(profile)
          return
        end

        render json: { errors: profile.errors.full_messages }, status: :unprocessable_entity
      end

      private

      def profile_payload(profile)
        {
          profile: profile.as_api_json,
          help_counts: profile.help_counts,
          appreciation: profile.appreciation_summary
        }
      end

      def profile_params
        root = params[:profile].is_a?(ActionController::Parameters) ? params[:profile] : params
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
