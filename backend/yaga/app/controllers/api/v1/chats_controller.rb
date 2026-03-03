require 'time'

module Api
  module V1
    class ChatsController < BaseController
      GUEST_SYSTEM_EMAIL = 'anonymous+guest@yaga.local'.freeze

      before_action :require_profile!
      before_action :set_chat, only: [:show, :messages, :create_message]

      def index
        conversations = ChatConversation.where(:participant_ids.in => [current_profile.id.to_s])
          .order_by(last_message_at: :desc, updated_at: :desc)
          .limit(100)

        render json: {
          conversations: conversations.map { |conversation| conversation.as_api_json_for(current_profile) }
        }
      end

      def create
        participant = target_participant
        return if performed?

        conversation = ChatConversation.find_or_create_direct!(current_profile.id, participant.id)
        messages = conversation.chat_messages.order_by(created_at: :asc).limit(message_limit)

        render json: conversation_payload(conversation, messages), status: :created
      rescue Mongoid::Errors::Validations => e
        render json: { errors: e.document.errors.full_messages }, status: :unprocessable_entity
      end

      def show
        messages = @chat.chat_messages.order_by(created_at: :asc).limit(message_limit)
        render json: conversation_payload(@chat, messages)
      end

      def messages
        chat_messages = @chat.chat_messages
        if params[:after].present?
          parsed_after = parse_time(params[:after])
          if parsed_after.nil?
            render json: { error: 'Invalid after timestamp. Use ISO8601 format.' }, status: :unprocessable_entity
            return
          end
          chat_messages = chat_messages.where(:created_at.gt => parsed_after)
        end

        chat_messages = chat_messages.order_by(created_at: :asc).limit(message_limit)
        render json: {
          messages: chat_messages.map { |message| message.as_api_json(current_profile) }
        }
      end

      def create_message
        chat_message = @chat.chat_messages.new(message_params)
        chat_message.sender = current_profile
        chat_message.sender_profile_id = current_profile.id.to_s

        if chat_message.save
          @chat.touch_last_message!(chat_message.body)
          render json: { message: chat_message.as_api_json(current_profile) }, status: :created
          return
        end

        render json: { errors: chat_message.errors.full_messages }, status: :unprocessable_entity
      end

      private

      def set_chat
        @chat = ChatConversation.where(id: params[:id]).first
        if @chat.nil?
          render json: { error: 'Conversation not found' }, status: :not_found
          return
        end

        return if @chat.include_profile?(current_profile.id)

        render json: { error: 'Forbidden' }, status: :forbidden
      end

      def target_participant
        participant_id = params[:participant_profile_id].to_s.strip
        if participant_id.blank?
          render json: { error: 'participant_profile_id is required' }, status: :unprocessable_entity
          return nil
        end

        if participant_id == current_profile.id.to_s
          render json: { error: 'Cannot start a chat with yourself' }, status: :unprocessable_entity
          return nil
        end

        participant = Profile.where(id: participant_id).first
        if participant.nil?
          render json: { error: 'Target user not found' }, status: :not_found
          return nil
        end

        if participant.userauth && participant.userauth.email == GUEST_SYSTEM_EMAIL
          render json: { error: 'Cannot chat with the anonymous guest profile' }, status: :unprocessable_entity
          return nil
        end

        participant
      end

      def message_params
        root = params[:message].is_a?(ActionController::Parameters) ? params[:message] : params
        root.permit(:body)
      end

      def message_limit
        requested = params[:limit].to_i
        return 80 if requested <= 0

        [requested, 250].min
      end

      def parse_time(value)
        Time.iso8601(value.to_s)
      rescue ArgumentError
        nil
      end

      def conversation_payload(conversation, messages)
        {
          conversation: conversation.as_api_json_for(current_profile),
          messages: messages.map { |message| message.as_api_json(current_profile) }
        }
      end
    end
  end
end
