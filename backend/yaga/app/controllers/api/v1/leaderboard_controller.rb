module Api
  module V1
    class LeaderboardController < BaseController
      GUEST_SYSTEM_EMAIL = 'anonymous+guest@yaga.local'.freeze

      skip_before_action :authenticate_api_user!, only: [:index]
      before_action :authenticate_api_user_if_present!, only: [:index]

      def index
        eligible_profiles = Profile.all.reject do |profile|
          profile.userauth && profile.userauth.email == GUEST_SYSTEM_EMAIL
        end

        rows = eligible_profiles.map do |profile|
          given_count = Help.where(helper_id: profile.id, status: 'completed').count
          taken_count = Help.where(requester_id: profile.id, status: 'completed').count
          contributed_count = Help.where(:status.in => %w[matched in_progress completed]).any_of(
            { requester_id: profile.id },
            { helper_id: profile.id }
          ).count
          appreciation_points = Appreciation.where(to_profile_id: profile.id).sum(:reward_points).to_i

          score = (
            (given_count * 5.0) +
            (taken_count * 2.0) +
            (contributed_count * 1.5) +
            (appreciation_points * 0.2)
          ).round(2)

          {
            profile: profile.as_api_json,
            score: score,
            helps_given: given_count,
            helps_taken: taken_count,
            helps_contributed: contributed_count,
            appreciation_points: appreciation_points
          }
        end

        sorted = rows.sort_by do |row|
          [
            -row[:score],
            -row[:helps_given],
            -row[:helps_contributed],
            row[:profile][:full_name].to_s
          ]
        end

        ranked = sorted.each_with_index.map do |row, index|
          row.merge(rank: index + 1)
        end

        current_rank = nil
        if current_profile.present?
          current_rank = ranked.find { |row| row[:profile][:id] == current_profile.id.to_s }&.dig(:rank)
        end

        render json: {
          leaderboard: ranked.first(100),
          current_user_rank: current_rank
        }
      end
    end
  end
end
