class AddPostFlagToRamens < ActiveRecord::Migration[5.1]
  def change
    add_column :ramen, :post_flag, :boolean, default: false
  end
end
