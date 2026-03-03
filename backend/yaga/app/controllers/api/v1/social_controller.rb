module Api
  module V1
    class SocialController < BaseController
      GUEST_SYSTEM_EMAIL = 'anonymous+guest@yaga.local'.freeze

      skip_before_action :authenticate_api_user!, only: [:graph]
      before_action :authenticate_api_user_if_present!, only: [:graph]
      before_action :require_profile!, only: [:follow, :unfollow]

      def graph
        target_profile = find_target_profile
        return if performed?

        unless target_profile.network_visible_to?(current_profile)
          render json: {
            error: 'This user has restricted their network visibility',
            network_visibility: target_profile.network_visibility
          }, status: :forbidden
          return
        end

        render json: graph_payload(target_profile)
      end

      def follow
        target_profile = follow_target_profile
        return if performed?

        if Follow.where(follower_profile_id: current_profile.id, followed_profile_id: target_profile.id).exists?
          render json: graph_payload(target_profile), status: :ok
          return
        end

        follow = Follow.new(follower: current_profile, followed: target_profile)
        if follow.save
          render json: graph_payload(target_profile), status: :created
          return
        end

        render json: { errors: follow.errors.full_messages }, status: :unprocessable_entity
      end

      def unfollow
        target_profile = follow_target_profile
        return if performed?

        relation = Follow.where(follower_profile_id: current_profile.id, followed_profile_id: target_profile.id).first
        if relation.nil?
          render json: graph_payload(target_profile), status: :ok
          return
        end

        relation.destroy
        render json: graph_payload(target_profile)
      end

      private

      def find_target_profile
        if params[:profile_id].present?
          profile = Profile.where(id: params[:profile_id].to_s).first
          if profile.nil?
            render json: { error: 'Profile not found' }, status: :not_found
            return nil
          end

          return profile
        end

        if current_profile.present?
          return current_profile
        end

        render json: { error: 'profile_id is required for guests' }, status: :unprocessable_entity
        nil
      end

      def follow_target_profile
        profile_id = params[:profile_id].to_s.strip
        if profile_id.blank?
          render json: { error: 'profile_id is required' }, status: :unprocessable_entity
          return nil
        end

        target_profile = Profile.where(id: profile_id).first
        if target_profile.nil?
          render json: { error: 'Profile not found' }, status: :not_found
          return nil
        end

        if target_profile.id == current_profile.id
          render json: { error: 'Cannot follow yourself' }, status: :unprocessable_entity
          return nil
        end

        if guest_system_profile?(target_profile)
          render json: { error: 'Cannot follow anonymous guest profile' }, status: :unprocessable_entity
          return nil
        end

        unless target_profile.profile_visible_to?(current_profile) || target_profile.profile_visibility == 'private'
          render json: { error: 'This profile is not visible based on user settings' }, status: :forbidden
          return nil
        end

        target_profile
      end

      def graph_payload(target_profile)
        follower_relationships = Follow.where(followed_profile_id: target_profile.id).order_by(created_at: :desc).limit(80).to_a
        following_relationships = Follow.where(follower_profile_id: target_profile.id).order_by(created_at: :desc).limit(80).to_a

        follower_ids = follower_relationships.map { |row| row.follower_profile_id.to_s }
        following_ids = following_relationships.map { |row| row.followed_profile_id.to_s }

        follower_profiles_by_id = Profile.where(:id.in => follower_ids).to_a.each_with_object({}) do |profile, memo|
          memo[profile.id.to_s] = profile
        end
        following_profiles_by_id = Profile.where(:id.in => following_ids).to_a.each_with_object({}) do |profile, memo|
          memo[profile.id.to_s] = profile
        end

        followers = follower_ids.map { |profile_id| follower_profiles_by_id[profile_id] }.compact.map do |profile|
          profile_node(profile)
        end
        following = following_ids.map { |profile_id| following_profiles_by_id[profile_id] }.compact.map do |profile|
          profile_node(profile)
        end

        viewer_follows_target = false
        target_follows_viewer = false
        can_follow = false
        can_unfollow = false

        if current_profile.present? && current_profile.id != target_profile.id
          viewer_follows_target = Follow.where(
            follower_profile_id: current_profile.id,
            followed_profile_id: target_profile.id
          ).exists?
          target_follows_viewer = Follow.where(
            follower_profile_id: target_profile.id,
            followed_profile_id: current_profile.id
          ).exists?
          can_follow = !viewer_follows_target
          can_unfollow = viewer_follows_target
        end

        {
          profile_id: target_profile.id.to_s,
          network_visibility: target_profile.network_visibility,
          followers: followers,
          following: following,
          follower_count: Follow.where(followed_profile_id: target_profile.id).count,
          following_count: Follow.where(follower_profile_id: target_profile.id).count,
          viewer_follows_target: viewer_follows_target,
          target_follows_viewer: target_follows_viewer,
          can_follow: can_follow,
          can_unfollow: can_unfollow
        }
      end

      def profile_node(profile)
        payload = profile.as_api_json
        payload.merge(
          headline: social_headline(profile)
        )
      end

      def social_headline(profile)
        user_type = profile.user_type || 'Individual'
        if profile.organization_name.present?
          return "#{user_type} | #{profile.organization_name}"
        end

        "#{user_type} Contributor"
      end

      def guest_system_profile?(profile)
        profile.userauth && profile.userauth.email == GUEST_SYSTEM_EMAIL
      end
    end
  end
end
