Rails.application.routes.draw do
  # static pages
  root 'home#top'
  get 'home/about'
  get 'home/contact'
  get '/auth/:provider/callback', to: 'sessions#create'
  get '/logout', to: 'sessions#destroy'
  resource :ramens
end
