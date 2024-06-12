import glob
import os
from pathlib import Path



if __name__ == "__main__":
    # 注意這邊生產出來的 command 是需要在正確位置上執行的，不是生出來就可以執行，需檢查各參數是否正確

    root_folder = r"./"

    # fetch all folder name by root_folder
    folders = glob.glob(root_folder + "/*")
    folders = [Path(_).name for _ in folders]
    # filter only folder's one
    folders = [f for f in folders if os.path.isdir(f)]

    # below is very hardcode case
    """
    example command as below:
     python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" 
     --local_video --local_video_path="./datacase/experiment4paper/{X_2_Y}/{VIDEO_NAME}.mkv" 
     --output_save_path="./datacase/experiment4paper/{X_2_Y}/{VIDEO_NAME}"
     
     so key just case: 資料夾名稱
     # 以及 {VIDEO_NAME} 下的所有 MKV 檔案名稱
    """

    for folder in folders:
        # FETCH ALL CURRENT FOLDER BELOWE ALL MKV FILES
        mkv_files = glob.glob(f"{folder}/*.mkv")
        for mkv_name in mkv_files:
            mkv_name = Path(mkv_name).stem
            print(f"\
python step_2_eat_pic_to_gen_mapping.py --case_root \"./datacase/case1\" \
--local_video --local_video_path=\"./datacase/experiment4paper/{folder}/{mkv_name}.mkv\" \
--seat_over_table_rate=1.3 \
--output_save_path=\"./datacase/experiment4paper_1_3/{folder}/{mkv_name}\"")
        # print command




    # TODO : run as below by 2024-06-11
    """
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716166626497_25833.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716166626497_25833"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716176433698_67499.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716176433698_67499"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716178963497_66001.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716178963497_66001"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716343016298_65998.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716343016298_65998"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716343212796_64500.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716343212796_64500"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716349106296_39300.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716349106296_39300"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716358625831_66000.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716358625831_66000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716359625830_69002.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716359625830_69002"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716430195298_63000.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716430195298_63000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716431975098_68999.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716431975098_68999"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716440951398_75000.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716440951398_75000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716446443031_50167.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716446443031_50167"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716447704532_40765.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716447704532_40765"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716448662530_61501.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716448662530_61501"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716520937064_66000.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716520937064_66000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716525920197_58299.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716525920197_58299"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716602004998_64232.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716602004998_64232"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716695009298_59100.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716695009298_59100"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716166243998_45733.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716166243998_45733"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716189403797_67499.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716189403797_67499"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716358625831_66000.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716358625831_66000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716365995598_74999.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716365995598_74999"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716430195298_63000.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716430195298_63000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716431975098_68999.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716431975098_68999"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716440951398_75000.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716440951398_75000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716442338898_32968.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716442338898_32968"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716447605529_26802.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716447605529_26802"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716520937064_66000.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716520937064_66000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716525920197_58299.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716525920197_58299"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716602004998_64232.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716602004998_64232"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716695009298_59100.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716695009298_59100"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716168757297_45600.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716168757297_45600"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716180647496_72001.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716180647496_72001"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716299569631_71999.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716299569631_71999"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716348879798_39367.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716348879798_39367"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716356564030_46267.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716356564030_46267"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716363265331_24299.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716363265331_24299"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716364980631_57267.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716364980631_57267"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716440032396_75001.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716440032396_75001"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716443879030_30267.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716443879030_30267"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716447484032_46598.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716447484032_46598"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716451196831_75001.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716451196831_75001"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716454338332_70500.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716454338332_70500"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716455737831_23900.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716455737831_23900"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716459900631_58766.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716459900631_58766"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716526739197_66001.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716526739197_66001"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716533170999_14699.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716533170999_14699"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716626899132_70500.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716626899132_70500"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716712824029_64503.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716712824029_64503"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716180957997_66433.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716180957997_66433"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716186104299_29833.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716186104299_29833"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716296683130_23002.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716296683130_23002"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716348292298_36266.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716348292298_36266"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716364621132_71897.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716364621132_71897"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716435695596_47569.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716435695596_47569"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716453554330_42568.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716453554330_42568"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716528988699_19900.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716528988699_19900"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716532121196_67501.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716532121196_67501"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716609100732_61865.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716609100732_61865"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716610057530_30467.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716610057530_30467"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716695929531_57665.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716695929531_57665"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716187755797_74999.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716187755797_74999"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716193517098_41199.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716193517098_41199"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716364980631_57267.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716364980631_57267"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716367090096_67502.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716367090096_67502"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716385338697_23768.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716385338697_23768"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716439658897_69967.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716439658897_69967"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716443879030_30267.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716443879030_30267"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716712824029_64503.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716712824029_64503"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716182290998_55032.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716182290998_55032"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716186242297_64900.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716186242297_64900"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716298065630_75000.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716298065630_75000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716442467531_51799.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716442467531_51799"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716453717832_38599.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716453717832_38599"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716529522197_66000.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716529522197_66000"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716532236696_20934.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716532236696_20934"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716613650530_64500.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716613650530_64500"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716615100530_67502.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716615100530_67502"
python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716708313330_73502.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716708313330_73502"

    """