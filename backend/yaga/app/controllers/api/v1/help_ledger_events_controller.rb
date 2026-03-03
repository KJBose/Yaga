module Api
  module V1
    class HelpLedgerEventsController < BaseController
      skip_before_action :authenticate_api_user!, only: [:index]
      before_action :authenticate_api_user_if_present!, only: [:index]
      before_action :set_help
      before_action :require_profile!, only: [:create, :update]
      before_action :ensure_help_participant!, only: [:create, :update]
      before_action :set_ledger_event, only: [:update]

      def index
        events = @help.ledger_events.order_by(created_at: :desc)
        render json: {
          ledger: @help.ledger_summary,
          chain: @help.chain_summary,
          ledger_events: events.map(&:as_api_json)
        }
      end

      def create
        event = @help.ledger_events.new(create_ledger_event_params)
        event.actor_profile = current_profile

        if event.save
          render json: {
            ledger_event: event.as_api_json,
            ledger: @help.reload.ledger_summary,
            chain: @help.chain_summary
          }, status: :created
          return
        end

        render json: { errors: event.errors.full_messages }, status: :unprocessable_entity
      end

      def update
        if @ledger_event.update_attributes(update_ledger_event_params)
          render json: {
            ledger_event: @ledger_event.as_api_json,
            ledger: @help.reload.ledger_summary,
            chain: @help.chain_summary
          }
          return
        end

        render json: { errors: @ledger_event.errors.full_messages }, status: :unprocessable_entity
      end

      private

      def set_help
        @help = Help.where(id: params[:help_id]).first
        return if @help.present?

        render json: { error: 'Help not found' }, status: :not_found
      end

      def set_ledger_event
        @ledger_event = @help.ledger_events.where(id: params[:id]).first
        return if @ledger_event.present?

        render json: { error: 'Ledger event not found' }, status: :not_found
      end

      def ensure_help_participant!
        return if current_profile.present? && @help.involve_profile?(current_profile)

        render json: { error: 'Only help participants can manage ledger events' }, status: :forbidden
      end

      def create_ledger_event_params
        root = params[:ledger_event].is_a?(ActionController::Parameters) ? params[:ledger_event] : params
        root.permit(
          :action,
          :source_type,
          :chain_state,
          :tx_hash,
          :chain_id,
          :contract_address,
          :block_number,
          :log_index,
          :wallet_address,
          :metadata_hash,
          :payload_hash,
          :onchain_help_id,
          :idempotency_key,
          :notes,
          :error_message
        )
      end

      def update_ledger_event_params
        root = params[:ledger_event].is_a?(ActionController::Parameters) ? params[:ledger_event] : params
        root.permit(
          :source_type,
          :chain_state,
          :tx_hash,
          :chain_id,
          :contract_address,
          :block_number,
          :log_index,
          :wallet_address,
          :metadata_hash,
          :payload_hash,
          :onchain_help_id,
          :idempotency_key,
          :notes,
          :error_message
        )
      end
    end
  end
end
