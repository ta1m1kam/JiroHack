require 'twitter'

class TwitterAPI
  attr_reader :client

  def initialize(current_user)
    @client =
      Twitter::REST::Client.new do |config|
        config.consumer_key        = ENV["TWITTER_API_KEY"]
        config.consumer_secret     = ENV["TWITTER_API_SECRET"]
        user_auth = current_user
        # config.access_token        = ENV["TWITTER_ACCESS_TOKEN"]
        # config.access_token_secret = ENV["TWITTER_ACCESS_SECRET"]
        config.access_token = user_auth.token
        config.access_token_secret = user_auth.secret
      end
  end

  def update(msg, img)
    client.update_with_media(msg, img)
  end
end
