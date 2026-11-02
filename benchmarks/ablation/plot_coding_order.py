import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 把Ocean-large一个slice的三种编码顺序画在Mollweide等面积投影上，颜色表示编码顺序中的位置
# 编码顺序由s1_compress -coding-order-output写出（单partition、单标签、不做坐标缩放），见ablation_work/coding_order/*/compress.log
project_root=Path(__file__).resolve().parents[2]
data_root=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data')
work_root=data_root/'ablation_work/coding_order'
output_dir=project_root/'results/ablation/coding_order'
orderings=(('global','Reverse construction (global, UNSZ default)'),('local','Interpolation-style rounds (local)'),('forward','Forward construction'))
snapshot_fractions=(0.001,0.01,0.1)


def oklab_to_linear_srgb(lightness,a,b):
    l=(lightness+.3963377774*a+.2158037573*b)**3;m=(lightness-.1055613458*a-.0638541728*b)**3;s=(lightness-.0894841775*a-1.2914855480*b)**3
    return np.array([4.0767416621*l-3.3077115913*m+.2309699292*s,-1.2684380046*l+2.6097574011*m-.3413193965*s,-.0041960863*l-.7034186147*m+1.7076147010*s])


def linear_to_srgb(linear):
    return np.where(linear<=.0031308,12.92*linear,1.055*np.maximum(linear,0)**(1/2.4)-.055)


def srgb_to_linear(srgb):
    return np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)


def isoluminant_colormap(lightness=.62,chroma=.16,hue_start=250,hue_end=410):
    # OKLCH中亮度和彩度固定，色相从蓝经紫、品红变到红。编码位置相差很大的点混在同一像素时，平均颜色的彩度降低、发灰
    colors=[]
    for hue in np.radians(np.linspace(hue_start,hue_end,256)):
        colors.append(linear_to_srgb(np.clip(oklab_to_linear_srgb(lightness,chroma*np.cos(hue),chroma*np.sin(hue)),0,1)))
    return matplotlib.colors.ListedColormap(colors,name='isoluminant')


def lon_lat(xyz):
    return np.arctan2(xyz[:,1],xyz[:,0]),np.arcsin(np.clip(xyz[:,2]/np.linalg.norm(xyz,axis=1),-1,1))


def mollweide(lon,lat):
    # 等面积投影：解2θ+sin2θ=π·sinφ，x∈[-2√2,2√2]，y∈[-√2,√2]
    double_theta=lat.copy()
    for _ in range(40):
        double_theta-=(double_theta+np.sin(double_theta)-np.pi*np.sin(lat))/np.maximum(1+np.cos(double_theta),1e-12)
    return 2*np.sqrt(2)/np.pi*lon*np.cos(double_theta/2),np.sqrt(2)*np.sin(double_theta/2)


def box_blur(image,radius):
    padded=np.pad(image,[(radius,radius),(radius,radius)]+[(0,0)]*(image.ndim-2))
    height,width=image.shape[:2]
    return sum(padded[dy:dy+height,dx:dx+width] for dy in range(2*radius+1) for dx in range(2*radius+1))/(2*radius+1)**2


def render_mean_color(x,y,colors,width=2000,blur_radius=1):
    # 把点分到像素里，每个像素取其中各点颜色在线性RGB中的平均。网格在两极比赤道密约9倍，取平均而不是叠加透明度，避免颜色浓淡反映网格密度
    height=width//2
    ix=np.clip(((x+2*np.sqrt(2))/(4*np.sqrt(2))*width).astype(np.int64),0,width-1)
    iy=np.clip(((np.sqrt(2)-y)/(2*np.sqrt(2))*height).astype(np.int64),0,height-1)
    pixel=iy*width+ix
    count=box_blur(np.bincount(pixel,minlength=width*height).reshape(height,width).astype(np.float64),blur_radius)
    linear=srgb_to_linear(colors)
    color_sum=box_blur(np.stack([np.bincount(pixel,weights=linear[:,channel],minlength=width*height).reshape(height,width) for channel in range(3)],axis=2),blur_radius)
    image=np.ones((height,width,4));image[...,3]=0
    ocean=count>.25
    image[ocean,:3]=linear_to_srgb(color_sum[ocean]/count[ocean][:,None]);image[ocean,3]=1
    return image


def draw_mollweide_frame(axis):
    for meridian in np.radians(np.arange(-180,181,30)):
        latitude=np.radians(np.linspace(-90,90,181))
        axis.plot(*mollweide(np.full_like(latitude,meridian),latitude),color='0.8',linewidth=.4,zorder=0 if abs(meridian)<np.pi else 3)
    for parallel in np.radians(np.arange(-60,61,30)):
        longitude=np.radians(np.linspace(-180,180,361))
        axis.plot(*mollweide(longitude,np.full_like(longitude,parallel)),color='0.8',linewidth=.4,zorder=0)
    axis.set_xlim(-2*np.sqrt(2)*1.01,2*np.sqrt(2)*1.01);axis.set_ylim(-np.sqrt(2)*1.01,np.sqrt(2)*1.01)
    axis.set_aspect('equal');axis.axis('off')


def style_axis(axis,title):
    axis.set_title(title,fontsize=11)
    axis.set_xticklabels([]);axis.set_yticklabels([])
    axis.grid(color='0.85',linewidth=.4)


def plot_paper_figure(x,y,lon_degree,lat_degree,orders,colormap,path):
    # 论文图：第一行是最先编码的1%的点，第二行是赤道太平洋6°×3°区域内各点的编码位置。布局以pt为单位，宽度为双栏全宽
    sys.path.insert(0,str(project_root/'third_party/paperfig/src'))
    from paperfig import PaperFigureSpec,apply_paper_style,save_paper_png_vertical_crop
    titles=('Reverse construction','Round-based order','Forward construction')
    with plt.rc_context():
        apply_paper_style()
        spec=PaperFigureSpec.text()
        # paperfig导出时使用正方形画布再裁掉上下空白，因此从画布顶部往下按pt排版
        width_pt,height_pt=spec.size_in[0]*72,spec.size_in[1]*72
        figure=plt.figure(figsize=spec.size_in,dpi=spec.dpi)
        side_margin,column_gap,panel_height,caption_height=4,6,84,13
        panel_width=(width_pt-2*side_margin-2*column_gap)/3
        def add_axis(column,top):
            return figure.add_axes(((side_margin+column*(panel_width+column_gap))/width_pt,(top-panel_height)/height_pt,panel_width/width_pt,panel_height/height_pt))
        background=render_mean_color(x,y,np.full((len(x),3),.88),width=1400)
        region=np.flatnonzero((lon_degree>-150)&(lon_degree<-144)&(lat_degree>-1.5)&(lat_degree<1.5))
        point_num=len(x)
        for column,(name,_) in enumerate(orderings):
            axis=add_axis(column,height_pt-12)
            axis.imshow(background,extent=(-2*np.sqrt(2),2*np.sqrt(2),-np.sqrt(2),np.sqrt(2)),interpolation='nearest',zorder=0)
            ids=orders[name][:point_num//100]
            axis.scatter(x[ids],y[ids],s=.12,color='#1f3f6e',linewidths=0,rasterized=True,zorder=1)
            axis.add_patch(matplotlib.patches.Ellipse((0,0),4*np.sqrt(2),2*np.sqrt(2),fill=False,edgecolor='.4',linewidth=.5,zorder=2))
            axis.set_xlim(-2*np.sqrt(2)*1.005,2*np.sqrt(2)*1.005);axis.set_ylim(-np.sqrt(2)*1.01,np.sqrt(2)*1.01);axis.set_xticks([]);axis.set_yticks([]);axis.axis('off')
            figure.text((side_margin+column*(panel_width+column_gap)+panel_width/2)/width_pt,(height_pt-12-panel_height-2)/height_pt,f'({"abc"[column]}) {titles[column]}',ha='center',va='top')
            axis=add_axis(column,height_pt-12-panel_height-caption_height-4)
            rank=np.empty(point_num);rank[orders[name]]=np.arange(point_num)
            region_rank=np.argsort(np.argsort(rank[region]))/len(region)
            axis.scatter(lon_degree[region],lat_degree[region],c=region_rank,cmap=colormap,vmin=0,vmax=1,s=4.5,linewidths=0)
            axis.set_xlim(-150,-144);axis.set_ylim(-1.5,1.5);axis.set_aspect('equal');axis.set_xticks([]);axis.set_yticks([])
            for spine in axis.spines.values():spine.set_linewidth(.5);spine.set_color('.4')
            figure.text((side_margin+column*(panel_width+column_gap)+panel_width/2)/width_pt,(height_pt-12-2*panel_height-caption_height-4-2)/height_pt,f'({"def"[column]}) {titles[column]}',ha='center',va='top')
        colorbar_axis=figure.add_axes(((width_pt-180)/2/width_pt,(height_pt-12-2*panel_height-2*caption_height-4-14)/height_pt,180/width_pt,5/height_pt))
        colorbar=figure.colorbar(plt.cm.ScalarMappable(cmap=colormap),cax=colorbar_axis,orientation='horizontal')
        colorbar.set_ticks([0,1],labels=['coded first','coded last']);colorbar.outline.set_linewidth(.5);colorbar_axis.tick_params(length=1.8,width=.45,pad=1.5)
        colorbar_axis.set_title('Position in the coding order within panels (d)--(f)'.replace('--','–'),pad=2)
        save_paper_png_vertical_crop(figure,path,spec)
        plt.close(figure)


def main():
    output_dir.mkdir(parents=True,exist_ok=True)
    xyz=np.fromfile(data_root/'Ocean-large/tensor_data/coords.dat',dtype=np.float32).reshape(-1,3).astype(np.float64)
    lon,lat=lon_lat(xyz)
    point_num=len(xyz)
    colormap=isoluminant_colormap()
    orders={name:np.fromfile(work_root/name/'coding_order.i32',dtype=np.int32) for name,_ in orderings}
    # 图1：全部点按编码位置着色，每个像素显示其中各点颜色的平均
    x,y=mollweide(lon,lat)
    figure,axes=plt.subplots(3,1,figsize=(9,13.5))
    for axis,(name,title) in zip(axes,orderings):
        rank=np.empty(point_num);rank[orders[name]]=np.arange(point_num)/point_num
        draw_mollweide_frame(axis)
        axis.imshow(render_mean_color(x,y,colormap(rank)[:,:3]),extent=(-2*np.sqrt(2),2*np.sqrt(2),-np.sqrt(2),np.sqrt(2)),interpolation='nearest',zorder=1)
        axis.add_patch(matplotlib.patches.Ellipse((0,0),4*np.sqrt(2),2*np.sqrt(2),fill=False,edgecolor='0.3',linewidth=.8,zorder=4))
        axis.set_title(title,fontsize=11)
    colorbar=figure.colorbar(plt.cm.ScalarMappable(cmap=colormap),ax=axes,orientation='horizontal',fraction=.03,pad=.02,aspect=40)
    colorbar.set_label('Position in coding order (0 = coded first, 1 = coded last)')
    figure.savefig(output_dir/'coding_order_rank.png',dpi=300,bbox_inches='tight')
    plt.close(figure)
    # 图2：编码顺序中最先编码的0.1%、1%和10%的点
    figure=plt.figure(figsize=(13.5,7.5))
    for row,(name,title) in enumerate(orderings):
        for column,fraction in enumerate(snapshot_fractions):
            ids=orders[name][:int(point_num*fraction)]
            axis=figure.add_subplot(3,3,row*3+column+1,projection='mollweide')
            axis.scatter(lon[ids],lat[ids],s=.3 if fraction<.01 else .05,color='#1f4e79',alpha=.8 if fraction<.01 else .25,linewidths=0,rasterized=True)
            style_axis(axis,f'{name}: first {fraction:.1%} coded')
    figure.savefig(output_dir/'coding_order_first_points.png',dpi=250,bbox_inches='tight')
    plt.close(figure)
    # 图3：赤道太平洋一个5°×5°的小区域，每个点不透明，颜色是它在区域内点中的编码位置
    lon_degree,lat_degree=np.degrees(lon),np.degrees(lat)
    region=np.flatnonzero((lon_degree>-150)&(lon_degree<-145)&(lat_degree>-2.5)&(lat_degree<2.5))
    figure,axes=plt.subplots(1,3,figsize=(13.5,4.8))
    for axis,(name,title) in zip(axes,orderings):
        rank=np.empty(point_num);rank[orders[name]]=np.arange(point_num)
        region_rank=np.argsort(np.argsort(rank[region]))/len(region)
        axis.scatter(lon_degree[region],lat_degree[region],c=region_rank,cmap=colormap,vmin=0,vmax=1,s=9,linewidths=0)
        axis.set_aspect('equal');axis.set_title(title,fontsize=10);axis.set_xlabel('longitude');axis.set_ylabel('latitude')
    colorbar=figure.colorbar(plt.cm.ScalarMappable(cmap=colormap),ax=axes,orientation='horizontal',fraction=.05,pad=.18,aspect=50)
    colorbar.set_label(f'Position in coding order among the {len(region)} points of this region')
    figure.savefig(output_dir/'coding_order_region.png',dpi=200,bbox_inches='tight')
    plt.close(figure)
    # 图4：论文用的组合图
    plot_paper_figure(x,y,lon_degree,lat_degree,orders,colormap,output_dir/'coding_order_paper.png')


if __name__=='__main__':
    main()
