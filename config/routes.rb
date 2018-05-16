Rails.application.routes.draw do
  # static pages
  root 'home#top'
  get 'home/about'
  get 'home/contact'
end
