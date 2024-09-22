import os
import shutil
from pathlib import Path

#
def classify_mkv_to_folder():
    root = 'C:\cgit\zjpj-lib-cam\datacase\experiment4paper'
    export_root = 'C:\cgit\zjpj-lib-cam\datacase\experiment4paper\export'

    # fetch all mkv path, base on the root
    mkv_list = []
    for cur_root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.mkv'):
                mkv_list.append(os.path.join(cur_root, file))

    # 使用列表解析獲取所有文件的 stem
    stem_list = [Path(file).stem for file in mkv_list]

    # 紀錄所有具有 mkv_list 名稱的資料夾
    folder_list = []
    for _ in Path(root).rglob('*'):
        if _.is_dir():
            if _.stem in stem_list:
                folder_list.append(_)

    print("以下資較夾，在對應目錄下有相對的 .mkv檔案。")
    for _ in folder_list:
        print(_)
        # build export path
        # split relative diff path by root
        diff_path = _.relative_to(root)
        print(diff_path)
        # mkdir path recursively
        tar_path = os.path.join(export_root, diff_path)
        os.makedirs(tar_path, exist_ok=True)
        # copy corresponding mkv file to tar_path
        for file in mkv_list:
            if Path(file).stem == _.stem:
                print("copy file: ", file, " to ", tar_path)
                # handle samefile error
                if os.path.exists(os.path.join(tar_path, Path(file).name)):
                    print("file exists, skip")
                    continue
                shutil.copy(file, tar_path)
        # break

if __name__ == "__main__":
    root = r'C:\cgit\zjpj-lib-cam\datacase\experiment4paper\export'
    sub_save_folder = 'fetch_frames'
    # 每5秒 擷取 mkv 檔案的一張圖片，直接存到對應的資料下，檔案名稱採用
    # {mkv_file_name}_{timestamp}.jpg

    # 1. 檢查所有資料夾下的 mkv 檔案
    mkv_list = []
    for cur_root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.mkv'):
                mkv_list.append(os.path.join(cur_root, file))

    # 2. 使用 ffmpeg 擷取圖片
    for mkv in mkv_list:
        _sub_folder = Path(mkv).parent.joinpath(sub_save_folder)
        os.makedirs(_sub_folder, exist_ok=True)
        # fps=1/5 , 每5秒擷取一張
        command = "ffmpeg -i", mkv, "-vf fps=1/5 ", os.path.join(_sub_folder, Path(mkv).stem +"_%03d.jpg")
        command = " ".join(command)
        #print(command)
        os.system(command)






    pass