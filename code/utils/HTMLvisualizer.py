import os

class HTMLVisualizer():
    def __init__(self, path):
        self.fn_html = open(path,"w")
        self.content = '''
<html lang="en">         <!-- 元素是页面的根元素 -->
<head>                     <!-- 元素包含文档的元数据 -->
    <meta charset="UTF-8">   <!-- 定义网页编码格式 -->
    <title>可视化结果</title>  <!-- 元素描述了文档的标题-->
</head>
	<style>
		#div1{
			width: 200px;height: 300px;/* 设置容器的宽高 */
			line-height: 300px; /*设置line-height与rongqi的height相等*/
			overflow: hidden; /*防止内容超出容器或者产生自动换行*/
			font-size: 20;
			display: inline-block;}
		#div2
		{
			width: 2500px;height: 300px;/* 设置容器的宽高 */
			line-height: 300px; /*设置line-height与rongqi的height相等*/
			overflow: hidden; /*防止内容超出容器或者产生自动换行*/
			display: inline-block;
			text-align:center;
			display: inline-block;
			}
		#div3{
			width: 1250px;height: 300px;/* 设置容器的宽高 */
			line-height: 300px; /*设置line-height与rongqi的height相等*/
			overflow: hidden; /*防止内容超出容器或者产生自动换行*/
			display: inline-block;
			text-align:center;
			display: inline-block;
			}
			
		#div4{
			width: 300px;height: 30px;/* 设置容器的宽高 */
			line-height: 30px; /*设置line-height与rongqi的height相等*/
			overflow: hidden; /*防止内容超出容器或者产生自动换行*/
			display: inline-block;
			text-align:center;
			}
		#div5{
			width: 200px;height: 30px;/* 设置容器的宽高 */
			line-height: 30px; /*设置line-height与rongqi的height相等*/
			overflow: hidden; /*防止内容超出容器或者产生自动换行*/
			font-size: 20;
			display: inline-block;}
		#div6
		{
			width: 2500px;height: 30px;/* 设置容器的宽高 */
			line-height: 30px; /*设置line-height与rongqi的height相等*/
			overflow: hidden; /*防止内容超出容器或者产生自动换行*/
			display: inline-block;
			text-align:center;
			display: inline-block;
			}
	}
	</style>
<body>'''


    def add_content(self,sample_id, mixed_mixture_path,video_frame_path,gt_audios_path,separated_sources_path,sources_loc_path,selected,est_video_audio_path):
        selected_str=[]
        for s in selected:
            if s:
                selected_str.extend(['style="color:green"',"selected"])
            else:
                selected_str.extend(['style="color:red"', "not_selected"])
        detected_objects_scripts= (
                '\t\t\t<img src="{}" alt="" width="300" height="300" style="margin-right:100;margin-left:100">\n' * len(video_frame_path)).format(
            *video_frame_path)
        gt_audios_scripts = (
                '\t\t\t<img src="{}" alt="" width="496" height="294">\n' * len(gt_audios_path)).format(
            *gt_audios_path)
        separated_sources_scripts = (
                    '\t\t\t<img src="{}" alt="" width="496" height="264">\n' * len(separated_sources_path)).format(
            *separated_sources_path)
        estimated_video_audio_scripts = (
                    '\t\t\t<img src="{}" alt="" width="496" height="294">  \n' * len(est_video_audio_path)).format(
            *est_video_audio_path)
        # sources_loc_scripts=('\t\t\t<img src="{}" alt="" width="300" height="300">\n' * len(separated_sources_path)).format(*sources_loc_path)
        # selected_str_scripts=('\t\t\t<div id="div4" {}> {}</div>\n'* len(separated_sources_path)).format(*selected_str)
        self.content += r'''
    <hr><!--定义水平线-->
	<h2>%d</h2>
    <br><!--换行-->
	<div id="div1">
      mix audio
    </div>
	<div id="div2">
      <img src="%s" alt="" width="496" height="264">
    </div> 
    <br><!--换行-->	<!--图像是通过标签 <img> 来定义的。 -->
	<div id="div1">
      detected objects
    </div>
	<div id="div2">
%s
    </div>
    <br><!--换行-->
	<div id="div1">
      gt audios
    </div>
    <div id="div2">
    %s
    </div>
	<br><!--换行-->
	<div id="div1">
      separated sources
    </div>
	<div id="div2">
%s
    </div>
	<br><!--换行-->
	<div id="div1">
      estmated video audios
    </div>
	<div id="div2">
%s
    </div>
'''%(sample_id, *mixed_mixture_path,detected_objects_scripts,gt_audios_scripts,separated_sources_scripts,estimated_video_audio_scripts)

    def write_html(self):
        self.content += '''</body>
</html>'''
        self.fn_html.write(self.content)
