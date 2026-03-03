module Api
  module V1
    class AppreciationsController < BaseController
      before_action :require_profile!, only: [:create]

      def index
        if current_profile.nil?
          render json: { appreciations: [] }
          return
        end

        scope = params[:scope].to_s
        appreciations = case scope
                        when 'given'
                          Appreciation.where(from_profile_id: current_profile.id)
                        when 'received'
                          Appreciation.where(to_profile_id: current_profile.id)
                        else
                          Appreciation.any_of(
                            { from_profile_id: current_profile.id },
                            { to_profile_id: current_profile.id }
                          )
                        end

        render json: { appreciations: appreciations.order_by(created_at: :desc).map(&:as_api_json) }
      end

      def create
        help = Help.where(id: appreciation_params[:help_id]).first

        if help.nil?
          render json: { error: 'Help not found' }, status: :not_found
          return
        end

        unless help.completed?
          render json: { error: 'Appreciation can be sent only after help is completed' }, status: :unprocessable_entity
          return
        end

        unless help.involve_profile?(current_profile)
          render json: { error: 'Only help participants can send appreciation' }, status: :forbidden
          return
        end

        recipient = resolve_recipient(help)
        if recipient.nil?
          render json: { error: 'Unable to determine appreciation recipient' }, status: :unprocessable_entity
          return
        end

        appreciation = Appreciation.new(
          help: help,
          from_profile: current_profile,
          to_profile: recipient,
          note: appreciation_params[:note],
          reward_points: appreciation_params[:reward_points]
        )

        if appreciation.save
          begin
            HelpLedgerEvent.create!(
              help: help,
              actor_profile: current_profile,
              action: 'appreciation',
              source_type: 'backend',
              chain_state: 'recorded',
              notes: "Reward points: #{appreciation.reward_points}"
            )
          rescue StandardError => e
            Rails.logger.warn("[HelpLedgerEvent] appreciation log failed for help=#{help.id}: #{e.class}: #{e.message}")
          end
          render json: { appreciation: appreciation.as_api_json }, status: :created
          return
        end

        render json: { errors: appreciation.errors.full_messages }, status: :unprocessable_entity
      end

      private

      def resolve_recipient(help)
        explicit_recipient = Profile.where(id: appreciation_params[:to_profile_id]).first if appreciation_params[:to_profile_id].present?
        return explicit_recipient if explicit_recipient.present?

        return help.helper if help.requester_id == current_profile.id

        help.requester
      end

      def appreciation_params
        root = params[:appreciation].is_a?(ActionController::Parameters) ? params[:appreciation] : params
        root.permit(:help_id, :to_profile_id, :note, :reward_points)
      end
    end
  end
end
