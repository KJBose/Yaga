class ApplicationController < ActionController::Base
  respond_to :html, :json
  before_action :set_cors_headers
  before_action :respond_to_preflight_request

  # Prevent CSRF attacks by raising an exception.
  # For APIs, you may want to use :null_session instead.
  #protect_from_forgery with: :exception
  protect_from_forgery with: :null_session

  private

  def set_cors_headers
    headers['Access-Control-Allow-Origin'] = '*'
    headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
    headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Accept, Authorization'
  end

  def respond_to_preflight_request
    return unless request.options?

    head :ok
  end
end
