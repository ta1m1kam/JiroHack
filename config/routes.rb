Rails.application.routes.draw do
  get 'cameras/new'

  # static pages
  root 'home#top'
  get 'home/about'
  get 'home/contact'
  get '/auth/:provider/callback', to: 'sessions#create'
  get '/logout', to: 'sessions#destroy'
  resource :cameras
end
