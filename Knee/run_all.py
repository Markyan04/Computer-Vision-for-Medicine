import os
import subprocess
import datetime
import sys

# 1. 定义你要遍历的文件夹和脚本名称
TARGET_FOLDERS = [
    # "HAM10000_MECS",
    # "chest-x-ray-image_MECS",
    # "Chest-Normal-Pneumonia_MECS",
    # "MRI_MECS"
    "../Knee"
]

TARGET_SCRIPTS = [
    # "ResNet_baseline.py",
    # "ResNet_baseline+WCE.py",
    # "ResNet_baseline+Loss4.py",
    # "ResNet_baseline+Loss5.py",
    # "ResNet_layer2+MDFA+CE.py",
    # "ResNet_layer2+MDFA+Loss4.py",

    # "ResNet_layer3+MDFA+CE.py",
    "ResNet_layer3+MDFA+Loss4.py",

    "ResNet_layer2+GCSA+CE.py",
    "ResNet_layer2+GCSA+Loss4.py",

    "ResNet_layer3+GCSA+CE.py",
    "ResNet_layer3+GCSA+Loss4.py",

    # "ResNet_layer2+MECS+CE.py",
    "ResNet_layer2+MECS+Loss4.py",

    # "ResNet_layer3+MECS+CE.py",
    # "ResNet_layer3+MECS+Loss4.py",

]

# 2. 定义存放所有独立日志的专属文件夹
LOG_DIR = "../Knee/logs"


def main():
    # 如果 logs 文件夹不存在，则创建它
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    start_time = datetime.datetime.now()
    print(f"\n批量实验启动时间: {start_time}\n" + "=" * 60)

    for folder in TARGET_FOLDERS:
        if not os.path.exists(folder):
            print(f"\n警告: 找不到文件夹 '{folder}'，已跳过。\n")
            continue

        for script in TARGET_SCRIPTS:
            script_path = os.path.join(folder, script)

            if not os.path.exists(script_path):
                print(f"\n警告: 找不到脚本 '{script_path}'，已跳过。\n")
                continue

            # 3. 动态生成每个实验专属的日志文件名
            # 去掉 '.py' 后缀，拼接成: logs/文件夹名_脚本名.log
            script_name_no_ext = os.path.splitext(script)[0]
            log_filename = f"{folder}_{script_name_no_ext}.log"
            log_filepath = os.path.join(LOG_DIR, log_filename)

            # 以写入模式(w)打开该专属日志文件
            with open(log_filepath, "w", encoding="utf-8") as log_file:
                # 记录单个实验开始
                run_header = f"\n{'-' * 60}\n"
                run_header += f"正在运行: 目录 [{folder}] -> 脚本 [{script}]\n"
                run_header += f"日志保存至: {log_filepath}\n"
                run_header += f"开始于: {datetime.datetime.now()}\n"
                run_header += f"{'-' * 60}\n"

                print(run_header, end="")
                log_file.write(run_header)
                log_file.flush()

                # 运行脚本
                process = subprocess.Popen(
                    [sys.executable, script],
                    cwd=folder,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8"
                )

                # 实时读取输出
                for line in process.stdout:
                    print(line, end="")
                    log_file.write(line)
                    log_file.flush()

                process.wait()

                # 记录单个实验结束
                run_footer = f"\n运行完毕: [{script}] | 退出码: {process.returncode}\n"
                run_footer += f"结束于: {datetime.datetime.now()}\n"
                print(run_footer)
                log_file.write(run_footer)
                log_file.flush()

    finish_msg = f"\n所有实验已执行完毕！所有日志文件均已保存在 '{LOG_DIR}' 文件夹中。\n"
    print(finish_msg)


if __name__ == "__main__":
    main()
