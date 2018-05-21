class CreateRamen < ActiveRecord::Migration[5.1]
  def change
    create_table :ramen do |t|
      t.string :image_url
      t.integer :score
      t.string :shop_name
      t.integer :user_id

      t.timestamps
    end

    add_index :ramen, :user_id
  end
end
