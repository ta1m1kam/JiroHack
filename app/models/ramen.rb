class Ramen < ApplicationRecord
  mount_uploader :image_url, ImageUrlUploader
  belongs_to :user, class_name: 'User'

  validates :shop_name, presence: true, length: { maximum: 20 }
  validates :image_url, presence: true
end
