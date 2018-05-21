class Ramen < ApplicationRecord
  mount_uploader :image_url, ImageUrlUploader
  belongs_to :user, class_name: 'User'
end
