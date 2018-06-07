class RamensController < ApplicationController
  before_action :authenticate

  def show
    @ramen = Ramen.find(params[:id])
    msg =  "#俺の二郎 ##{@ramen.shop_name} #ましましクーポン"
    # binding.pry
    unless @ramen.post_flag
      TwitterAPI.new.update(msg, "public#{@ramen.image_url.url}")
      @ramen.update(post_flag: true)
    end
  end

  def new
    @ramen = current_user.created_ramens.build
  end

  def create
    @ramen = current_user.created_ramens.build(ramen_params)
    if @ramen.save
      redirect_to ramen_url(id: @ramen.id), notice: '測定します。'
    else
      render :new
    end
  end

  private
    def ramen_params
      params.require(:ramen).permit(
        :image_url, :shop_name
      )
    end
end
