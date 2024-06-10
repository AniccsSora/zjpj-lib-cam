import glob
from pathlib import Path
import copy
import os

def file_exists_in_directory(root_dir, filename):
    """
    遞迴搜索指定目錄及其所有子目錄，檢查指定檔案是否存在。
    :param root_dir: 要搜索的根目錄
    :param filename: 要查找的檔案名
    :return: 如果找到檔案，返回 True，否則返回 False
    """
    # 使用 os.walk() 遍歷目錄
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 檢查當前目錄中的檔案列表是否包含目標檔案
        if filename in filenames:
            return True, os.path.join(dirpath, filename)  # 如果找到檔案，返回 True
    return False, None  # 遍歷完所有目錄後，如果沒有找到檔案，返回 False


def check_X_2_Y_case_number():
    # 根據此資料夾下的 folder name 當作 計算標題
    root_folder = r"C:\cgit\zjpj-lib-cam\datacase\experiment4paper"

    # fetch all folder name
    folders = glob.glob(root_folder + "/*")

    # list of name
    names = [Path(_).name for _ in folders]

    # gen list dict for each folder
    folder_dict = {}

    for name in names:
        new_list = []
        # 組合完整的資料夾路徑
        full_path = Path(root_folder).joinpath(name)
        # 尋找資料夾中所有的 mkv 檔案
        mkv_files = [_ for _ in full_path.glob('*.mkv')]
        new_list = copy.copy(mkv_files)

        folder_dict[name] = new_list
        # 輸出結果
        print(f'{name} has {len(mkv_files)} mkv files.')

    return folder_dict


if __name__ == "__main__":

    # 1
    all_filted_mkv = check_X_2_Y_case_number()

    # 2, 用 filted 的 mkv 檔案名稱，找是否有在以下 search_base 中。
    # 有的話，把 search_base 記錄下來。
    search_base = r"J:\yzu_lib_cam_save"
    #
    original_base = {}

    for name, mkv_files in all_filted_mkv.items():
        _original_path = []
        for mkv_file in mkv_files:
            mkv_file = Path(mkv_file)
            # search
            is_exists, _path_name = file_exists_in_directory(search_base, mkv_file.name)
            if is_exists:
                _original_path.append(_path_name)
            else:
                # 應存在才對，不應該找不到，因為是從來源資料夾複製而來的。
                _msg = f'{mkv_file.name} not found in {search_base}'
                raise Exception("not found should exists files!!! \n    ", _msg)

        #
        original_base[name] = _original_path

    # cals all file numbers
    _total = 0
    for name, mkv_files in all_filted_mkv.items():
        _total += len(mkv_files)
    print("total:", _total)

    print("===")

    # dump key and respond path to txt
    _cnt1 = 0  # total
    with open("X_2_Y_case_number.txt", "w", encoding='utf-8') as f:
        for key, value in original_base.items():
            f.write(f'{key}, len={len(value)}\n')
            _cnt1 += len(value)
            for _ in value:
                f.write(f'  {_}\n')
            f.write("\n")
        #
        f.write(f"total mkvs: {_cnt1}")