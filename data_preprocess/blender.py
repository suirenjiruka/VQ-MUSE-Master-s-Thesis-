# bvh_render.py
import bpy
import random
import sys, os
from mathutils import Vector
# 從 CLI 取得參數（在 -- 之後）


bvh_path =  "C:/Users/USER/Desktop/Tzu-Hsuan/SnapMoGen/renamed_bvhs/"
#fbx_filepath = "C:/Users/USER/Desktop/Tzu-Hsuan/human_base.fbx"
obj_filepath = "C:/Users/USER/Desktop/Tzu-Hsuan/human_base1.obj"
out_path  =  "C:/Users/USER/Desktop/Tzu-Hsuan/mogen_video/"

# 清空場景

def get_override_context():
    """
    尋找一個3D視圖區域，並為操作符創建一個有效的上下文覆蓋字典。
    """
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return {
                            'window': window,
                            'screen': screen,
                            'area': area,
                            'region': region,
                            'scene': bpy.context.scene,
                        }
    return {'scene': bpy.context.scene}

def auto_skin_bvh_to_model():
    """
    自動將場景中的 BVH 骨架綁定到一個模型上。
    """  

    # 1. 識別場景中的骨架和模型
    armature_obj = None
    mesh_obj = None

    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature_obj = obj
        elif obj.type == 'MESH':
            mesh_obj = obj
            
    if not armature_obj or not mesh_obj:
        print("錯誤：場景中必須同時包含一個骨架和一個模型。")
        return

    print(f"找到骨架: '{armature_obj.name}'")
    print(f"找到模型: '{mesh_obj.name}'")
    
     #Manual alignment
    armature_obj.location[2] = -0.35
    armature_obj.location[1] = -0.3
    armature_obj.scale[0] = 0.95
    armature_obj.scale[1] = 1.1
    armature_obj.scale[2] = 0.99
    mesh_obj.scale[0] = 0.475
    mesh_obj.scale[1] = 0.5

    # 2. (可選但推薦) 清除骨架姿態，恢復到 T-Pose
    if armature_obj.animation_data:
        temp_animation = armature_obj.animation_data.action
        armature_obj.animation_data.action = None
        
        
    mesh_obj.parent = None
    for group in mesh_obj.vertex_groups:
        mesh_obj.vertex_groups.remove(group)
    
    bpy.context.view_layer.objects.active = armature_obj
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        bpy.ops.object.mode_set(mode='OBJECT')
    
    print("骨架已重置到靜態姿勢 (Rest Pose)。")

    # 3. (可選但推薦) 應用模型的變換，確保縮放和旋轉是乾淨的
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mesh_obj.select_set(False)
    print(f"已應用 '{mesh_obj.name}' 的變換。")

    print("步驟 3: 修復模型幾何 (合併頂點 & 重算法線)...")
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    
    # 進入編輯模式
    bpy.ops.object.mode_set(mode='EDIT')
    # 合併重疊頂點
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    # 重新計算外部法線
    bpy.ops.mesh.normals_make_consistent(inside=False)
    # 返回物件模式
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')

    # 4. 執行綁定 (最關鍵的步驟)
    print("開始自動權重綁定...")
    
    # 取消選擇所有物體
    bpy.ops.object.select_all(action='DESELECT')
    
    # 選擇模型，然後選擇骨架
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    armature_obj.hide_set(True)
    bpy.context.view_layer.objects.active = armature_obj # 讓骨架成為活動物體
    
    for fcurve in temp_animation.fcurves:
        if "ROOT" == fcurve.data_path.split('"')[1]:
            print(fcurve.data_path)
    
    try:
        # 執行 "With Automatic Weights" 操作
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        bpy.ops.object.select_all(action='DESELECT')        
    except RuntimeError as e:
        print(f"綁定失敗: {e}")
        print("這通常意味著模型和骨架沒有正確對齊，或者模型拓撲有問題。")
    armature_obj.animation_data.action = temp_animation
    print(temp_animation.fcurves[0].evaluate(3)*0.95, temp_animation.fcurves[1].evaluate(3) * 1.1, temp_animation.fcurves[2].evaluate(3) *0.99 )
    armature_obj.location -=  Vector((temp_animation.fcurves[0].evaluate(3)*0.95 , -temp_animation.fcurves[2].evaluate(3) *1.1, temp_animation.fcurves[1].evaluate(3) * 0.99))

    
print("開始執行BVH導入腳本...")

dir = os.listdir(bvh_path)
for item in dir:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if item.lower().endswith(".bvh"):
        if not os.path.exists(bvh_path + item):
            print(f"錯誤：找不到文件，路徑為: {bvh_path}")
        else:
            override = get_override_context()

        # 3. 再次執行有問題的命令
        try:
            with bpy.context.temp_override(**override):
                bpy.ops.import_anim.bvh(
                    filepath=bvh_path + item,
                    global_scale = 0.05,
                    axis_forward='-Z',
                    axis_up='Y'
                )
                #bpy.ops.import_scene.fbx(filepath=fbx_filepath, global_scale = 0.5)
                bpy.ops.wm.obj_import(filepath=obj_filepath, global_scale = 0.493)
                auto_skin_bvh_to_model()
        except:
            continue            

        # 建立或定位 Camera

        C= random.randint(0, 16)

        if 'Camera' not in bpy.data.objects:
            if C == 0: 
                t = (0, 25, 2.6)
                r = (1.56, 0, 3.14)
            elif C==1:
                t = (16, 16, 2.6)
                r = (1.56, 0, 2.45)
            elif C==2:
                t = (-16, 16, 2.47)
                r = (1.56, 0, -2.45)
            elif C==3 or C==8 or C==10:
                t = (16, -16, 2.6)
                r = (1.56, 0, 0.7)
            elif C==4 or C==9 or C==11:
                t = (-16, -16, 2.6)
                r = (1.56, 0, -0.7)
            elif C==5:
                t = (14, -20, 3.8)
                r = (1.4, 0, 0.4)
            elif C==6:
                t = (-14, -20, 3.8)
                r = (1.4, 0, -0.4)
            else:
                t = (0, -25, 2.6)
                r = (1.56, 0, 0)
            cam_data = bpy.data.cameras.new('Camera') 
            cam_obj = bpy.data.objects.new('Camera', cam_data) 
            bpy.context.collection.objects.link(cam_obj) 
            cam_obj.location = t
            cam_obj.rotation_euler = r 
            
        with bpy.context.temp_override(**override):
            bpy.ops.view3d.view_camera()

        # 簡單建立光源
        light_data = bpy.data.lights.new(name="Light", type='AREA')
        light_obj = bpy.data.objects.new("Light", light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = (2, -3, 5)

        # Render 設定
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE_NEXT' # 建議明確指定渲染引擎 (CYCLES 或 EEVEE)
        scene.cycles.device = 'GPU'    # 如果有GPU，可以啟用

        scene.render.resolution_x = 1280
        scene.render.resolution_y = 720
        scene.render.fps = 30

        # 這裡的 frame_start 和 frame_end 應該要設定一個具體的值
        # 否則它會使用場景當前的值，可能是1
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                action = obj.animation_data.action
                fcurves = action.fcurves
                max_frame = float('-inf')
                
                for fc in fcurves:
                    keyframes = [kp.co.x for kp in fc.keyframe_points]
                    if keyframes:
                        max_frame = max(max_frame, max(keyframes))
                break
                
        scene.frame_start = 1 #scene.frame_start
        scene.frame_end = int(max_frame) # 假設渲染100幀

        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        scene.render.filepath = out_path + item[:-4] + ".mp4"
        scene.render.ffmpeg.constant_rate_factor = 'MEDIUM' # HIGH 是比較低的品質，MEDIUM 或 LOW 是更高品質
        scene.render.ffmpeg.video_bitrate = 6000

        # 開始 render 動畫
        print(f"開始渲染動畫，輸出到: {out_path + item}")
        override["scene"] = scene
        override = get_override_context()
        with bpy.context.temp_override(**override):
            bpy.ops.render.opengl(write_still=True, view_context=True, animation = True)
