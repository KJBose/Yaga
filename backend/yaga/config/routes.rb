Rails.application.routes.draw do
  devise_for :userauths
  #devise_for :users
  resources :users

  namespace :api do
    namespace :v1 do
      post 'auth/signup', to: 'auth#signup'
      post 'auth/signin', to: 'auth#signin'
      delete 'auth/signout', to: 'auth#signout'
      get 'auth/me', to: 'auth#me'

      resource :profile, only: [:show, :update]
      get 'profiles/:id', to: 'profiles#public_show'
      resources :helps, only: [:index, :show, :create, :update] do
        collection do
          get :prioritized
          get :my
        end

        resources :ledger_events, only: [:index, :create, :update], controller: 'help_ledger_events'

        member do
          post :take
          post :mark_in_progress
          post :complete
          post :cancel
        end
      end
      resources :chats, only: [:index, :show, :create] do
        member do
          get 'messages', to: 'chats#messages'
          post 'messages', to: 'chats#create_message'
        end
      end
      get 'social/graph', to: 'social#graph'
      post 'social/follow', to: 'social#follow'
      delete 'social/follow/:profile_id', to: 'social#unfollow'
      resources :appreciations, only: [:index, :create]
      get 'transformations', to: 'transformations#index'
      get 'leaderboard', to: 'leaderboard#index'
    end
  end

  match '*path', via: :options, to: proc {
    cors_headers = {
      'Access-Control-Allow-Origin' => '*',
      'Access-Control-Allow-Methods' => 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
      'Access-Control-Allow-Headers' => 'Origin, Content-Type, Accept, Authorization',
      'Access-Control-Max-Age' => '86400'
    }

    [204, cors_headers, []]
  }
  # The priority is based upon order of creation: first created -> highest priority.
  # See how all your routes lay out with "rake routes".

  # You can have the root of your site routed with "root"
  # root 'welcome#index'

  # Example of regular route:
  #   get 'products/:id' => 'catalog#view'

  # Example of named route that can be invoked with purchase_url(id: product.id)
  #   get 'products/:id/purchase' => 'catalog#purchase', as: :purchase

  # Example resource route (maps HTTP verbs to controller actions automatically):
  #   resources :products

  # Example resource route with options:
  #   resources :products do
  #     member do
  #       get 'short'
  #       post 'toggle'
  #     end
  #
  #     collection do
  #       get 'sold'
  #     end
  #   end

  # Example resource route with sub-resources:
  #   resources :products do
  #     resources :comments, :sales
  #     resource :seller
  #   end

  # Example resource route with more complex sub-resources:
  #   resources :products do
  #     resources :comments
  #     resources :sales do
  #       get 'recent', on: :collection
  #     end
  #   end

  # Example resource route with concerns:
  #   concern :toggleable do
  #     post 'toggle'
  #   end
  #   resources :posts, concerns: :toggleable
  #   resources :photos, concerns: :toggleable

  # Example resource route within a namespace:
  #   namespace :admin do
  #     # Directs /admin/products/* to Admin::ProductsController
  #     # (app/controllers/admin/products_controller.rb)
  #     resources :products
  #   end
end
