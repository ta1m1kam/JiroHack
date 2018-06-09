class User < ApplicationRecord
  has_many :created_ramens, class_name: 'Ramen', foreign_key: :user_id

  def self.find_or_create_from_auth_hash(auth_hash)
    provider = auth_hash[:provider]
    uid = auth_hash[:uid]
    nickname = auth_hash[:info][:nickname]
    image_url = auth_hash[:info][:image]
    token = auth_hash.credentials.token
    secret = auth_hash.credentials.secret

    User.find_or_create_by(provider: provider, uid: uid) do |user|
      user.nickname = nickname
      user.image_url = image_url
      user.token = token
      user.secret = secret
    end
  end
end
