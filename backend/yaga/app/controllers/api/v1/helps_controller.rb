require 'securerandom'

module Api
  module V1
    class HelpsController < BaseController
      GUEST_SYSTEM_EMAIL = 'anonymous+guest@yaga.local'.freeze

      skip_before_action :authenticate_api_user!, only: [:index, :prioritized, :show, :create]
      before_action :authenticate_api_user_if_present!, only: [:create]
      before_action :require_profile!, only: [:my, :update, :take, :mark_in_progress, :complete, :cancel]
      before_action :set_help, only: [:show, :update, :take, :mark_in_progress, :complete, :cancel]

      def index
        helps = filtered_helps.order_by(priority_score: :desc, created_at: :desc)
        render json: { helps: helps.map(&:as_api_json) }
      end

      def prioritized
        helps = Help.where(:status.in => %w[open matched in_progress]).order_by(priority_score: :desc, created_at: :desc)
        render json: { helps: helps.map(&:as_api_json) }
      end

      def my
        helps = Help.any_of(
          { requester_id: current_profile.id },
          { helper_id: current_profile.id }
        ).order_by(updated_at: :desc)

        render json: { helps: helps.map(&:as_api_json) }
      end

      def show
        render json: { help: @help.as_api_json }
      end

      def create
        help = Help.new(help_params)
        if current_profile.present?
          help.requester = current_profile
          help.anonymous_requester = false if help.anonymous_requester.nil?
        else
          help.requester = anonymous_requester_profile
          help.help_type = 'request'
          help.status = 'open'
          help.anonymous_requester = true
          if help.requester_name.blank?
            help.requester_name = 'Anonymous Guest'
          end
        end

        if help.save
          record_backend_ledger_event!(help, 'create')
          render json: { help: help.as_api_json }, status: :created
          return
        end

        render json: { errors: help.errors.full_messages }, status: :unprocessable_entity
      end

      def update
        unless @help.requester_id == current_profile.id
          render json: { error: 'Only requester can update help' }, status: :forbidden
          return
        end

        if @help.status != 'open'
          render json: { error: 'Only open helps can be edited' }, status: :unprocessable_entity
          return
        end

        if @help.update_attributes(help_params)
          record_backend_ledger_event!(@help, 'update')
          render json: { help: @help.as_api_json }
          return
        end

        render json: { errors: @help.errors.full_messages }, status: :unprocessable_entity
      end

      def take
        if @help.requester_id == current_profile.id
          render json: { error: 'Requester cannot take own help' }, status: :unprocessable_entity
          return
        end

        if @help.helper_id.present? && @help.helper_id != current_profile.id
          render json: { error: 'Help is already taken by another user' }, status: :unprocessable_entity
          return
        end

        @help.helper = current_profile
        @help.status = 'matched' if @help.status == 'open'

        if @help.save
          record_backend_ledger_event!(@help, 'take')
          render json: { help: @help.as_api_json }
          return
        end

        render json: { errors: @help.errors.full_messages }, status: :unprocessable_entity
      end

      def mark_in_progress
        unless actor_involved?(@help)
          render json: { error: 'Only participants can update status' }, status: :forbidden
          return
        end

        @help.transition_to!('in_progress')
        record_backend_ledger_event!(@help, 'mark_in_progress')
        render json: { help: @help.as_api_json }
      rescue Mongoid::Errors::Validations => e
        render json: { errors: e.document.errors.full_messages }, status: :unprocessable_entity
      end

      def complete
        unless actor_involved?(@help)
          render json: { error: 'Only participants can complete help' }, status: :forbidden
          return
        end

        @help.transition_to!('completed')
        record_backend_ledger_event!(@help, 'complete')
        render json: { help: @help.as_api_json }
      rescue Mongoid::Errors::Validations => e
        render json: { errors: e.document.errors.full_messages }, status: :unprocessable_entity
      end

      def cancel
        unless actor_involved?(@help)
          render json: { error: 'Only participants can cancel help' }, status: :forbidden
          return
        end

        @help.transition_to!('cancelled')
        record_backend_ledger_event!(@help, 'cancel')
        render json: { help: @help.as_api_json }
      rescue Mongoid::Errors::Validations => e
        render json: { errors: e.document.errors.full_messages }, status: :unprocessable_entity
      end

      private

      def set_help
        @help = Help.where(id: params[:id]).first

        if @help.nil?
          render json: { error: 'Help not found' }, status: :not_found
          return
        end
      end

      def actor_involved?(help)
        help.requester_id == current_profile.id || help.helper_id == current_profile.id
      end

      def filtered_helps
        helps = Help.all
        helps = helps.where(status: params[:status]) if params[:status].present?
        helps = helps.where(help_type: params[:help_type]) if params[:help_type].present?
        helps = helps.where(category: params[:category]) if params[:category].present?
        if params[:profile_id].present?
          helps = helps.any_of(
            { requester_id: params[:profile_id].to_s },
            { helper_id: params[:profile_id].to_s }
          )
        end
        helps
      end

      def help_params
        root = params[:help].is_a?(ActionController::Parameters) ? params[:help] : params
        root.permit(
          :title,
          :description,
          :category,
          :help_type,
          :status,
          :anonymous_requester,
          :requester_name,
          :requester_contact,
          :urgency,
          :scale,
          :required_resources,
          :time_available_hours,
          :supporters,
          :media_url,
          :acknowledgement_url
        )
      end

      def anonymous_requester_profile
        userauth = Userauth.where(email: GUEST_SYSTEM_EMAIL).first
        if userauth.nil?
          password = SecureRandom.hex(24)
          userauth = Userauth.create!(
            email: GUEST_SYSTEM_EMAIL,
            password: password,
            password_confirmation: password
          )
        end

        profile = userauth.profile
        if profile.nil?
          profile = userauth.create_profile(
            first_name: 'Anonymous',
            last_name: 'Guest',
            user_type: 'Individual',
            bio: 'System guest placeholder profile'
          )
          profile.save! unless profile.persisted?
        end

        profile
      end

      def record_backend_ledger_event!(help, action)
        actor = current_profile || help.requester
        HelpLedgerEvent.create!(
          help: help,
          actor_profile: actor,
          action: action,
          source_type: 'backend',
          chain_state: 'recorded',
          notes: 'Captured by backend lifecycle transition'
        )
      rescue StandardError => e
        Rails.logger.warn("[HelpLedgerEvent] create failed for help=#{help.id} action=#{action}: #{e.class}: #{e.message}")
      end
    end
  end
end
