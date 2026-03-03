module Api
  module V1
    class TransformationsController < BaseController
      def index
        if current_profile.nil?
          render json: {
            totals: {
              completed: 0,
              led: 0,
              contributed: 0
            },
            transformations: []
          }
          return
        end

        completed_helps = Help.where(status: 'completed').any_of(
          { requester_id: current_profile.id },
          { helper_id: current_profile.id }
        ).order_by(completed_at: :desc)

        render json: {
          totals: {
            completed: completed_helps.count,
            led: completed_helps.where(requester_id: current_profile.id).count,
            contributed: completed_helps.where(helper_id: current_profile.id).count
          },
          transformations: completed_helps.map { |help| transformation_payload(help) }
        }
      end

      private

      def transformation_payload(help)
        {
          help_id: help.id.to_s,
          title: help.title,
          category: help.category,
          completed_at: help.completed_at,
          priority_score: help.priority_score,
          requester: help.requester&.as_api_json,
          helper: help.helper&.as_api_json,
          appreciations: help.appreciations.map(&:as_api_json)
        }
      end
    end
  end
end
