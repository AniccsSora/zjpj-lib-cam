from pathlib import Path
import glob
import json



if __name__ == "__main__":
    detect_folder_name = [
                            "A_2_B",
                            "A_2_C",
                            "B_2_A",
                            "B_2_C",
                            "C_2_A",
                            "C_2_B",
                            ]
    # command run root. (in real use command dokoro.)
    command_run_root = Path(r"C:\cgit\zjpj-lib-cam")
    # this folder if enter, will find A_2_B, B_2_A, A_2_B_2_A, B_2_A_2_B ... folder name
    root_path =  Path(r"C:\cgit\zjpj-lib-cam\datacase\experiment4paper")  # this folder include A_2_B, B_2_A, A_2_B_2_A, B_2_A_2_B ... folder name
    # almostly fixed
    __case_root = "./datacase/case1"

    # output case name, base on command_run_root
    OUTPUT_RELATIVE_ROOT_PATH = "./"+root_path.joinpath("runs_output").relative_to(command_run_root).as_posix()
    #

    CASE_ROOT_ = "./datacase/case1"

    for p in detect_folder_name:
        if not root_path.joinpath(p).is_dir():
            raise FileNotFoundError(f"{p} not found in {root_path}")

    #dict_each_case_commands = { _ for _ in detect_folder_name: []}
    dict_each_case_commands = { _ : [] for _ in detect_folder_name}


    for key, _ in dict_each_case_commands.items():
        # dict_each_case_commands[key]

        _videos_path = glob.glob(str(root_path.joinpath(key, "*.mkv")))
        _videos_path = ["./"+str(Path(_).relative_to(command_run_root).as_posix()) for _ in _videos_path]
        for _video_path in _videos_path:
            _command = f'python step_2_eat_pic_to_gen_mapping.py --case_root "{CASE_ROOT_}" --local_video --local_video_path="{_video_path}" --output_save_path="{OUTPUT_RELATIVE_ROOT_PATH}/{key}/{Path(_video_path).stem}"'
            dict_each_case_commands[key].append(_command)


    # memory each case videos number
    dict_each_case_videos_number = { _ : len(__) for _, __ in dict_each_case_commands.items()}

    # printf(" type # each case )
    with open("./test_command_generator_output.txt", "w") as f:
        for key, _ in dict_each_case_commands.items():
            print(f":: #  {key}")
            for __ in _:
                print(__)
            print("\n")
            # mem
            dict_each_case_videos_number[key] = len(_)
            f.write(f":: #  {key}\n")
            for __ in _:
                f.write(__+"\n")
            f.write("\n")
    print("================================")
    for key, _ in dict_each_case_videos_number.items():
        print(f":: {key} : {_} videos")
    print("Please check \"test_command_generator_output.txt\"")