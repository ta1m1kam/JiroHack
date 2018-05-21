class RamensController < ApplicationController
  before_action :authenticate

  def show
    @ramen = Ramen.find(params[:id])
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
