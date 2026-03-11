import os
import sys
import subprocess
from datetime import datetime


def run_script_collect_output(script_path):
    print(f"\n{'=' * 80}")
    print(f"Starting: {os.path.basename(script_path)}")
    print(f"{'=' * 80}\n")

    start_time = datetime.now()

    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )

    output_lines = []

    # 实时打印到控制台，同时先缓存下来
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    process.wait()

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\nFinished: {os.path.basename(script_path)} | Exit code: {process.returncode}")
    print(f"Duration: {duration}\n")

    return {
        "script_name": os.path.basename(script_path),
        "script_path": script_path,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "return_code": process.returncode,
        "output": "".join(output_lines),
    }


def main():
    scripts = [
        "ResNet+MDFA_layer2.py",
        "ResNet+MDFA_layer3.py",
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(base_dir, f"batch_run_log_{timestamp}.txt")

    all_results = []

    for script in scripts:
        script_path = os.path.join(base_dir, script)

        if not os.path.exists(script_path):
            print(f"[ERROR] Script not found: {script_path}")
            all_results.append({
                "script_name": script,
                "script_path": script_path,
                "start_time": None,
                "end_time": None,
                "duration": None,
                "return_code": -999,
                "output": f"[ERROR] Script not found: {script_path}\n",
            })
            continue

        result = run_script_collect_output(script_path)
        all_results.append(result)

        # 每个文件跑完后，统一把该文件全部输出写进总日志
        with open(log_filename, "a", encoding="utf-8") as log_file:
            log_file.write("\n" + "=" * 100 + "\n")
            log_file.write(f"Script      : {result['script_name']}\n")
            log_file.write(f"Path        : {result['script_path']}\n")
            log_file.write(f"Start time  : {result['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"End time    : {result['end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Duration    : {result['duration']}\n")
            log_file.write(f"Exit code   : {result['return_code']}\n")
            log_file.write("-" * 100 + "\n")
            log_file.write("Console Output:\n")
            log_file.write("-" * 100 + "\n")
            log_file.write(result["output"])
            if not result["output"].endswith("\n"):
                log_file.write("\n")
            log_file.write("=" * 100 + "\n")

    # 最后再写一个总 summary
    with open(log_filename, "a", encoding="utf-8") as log_file:
        log_file.write("\n\nSUMMARY\n")
        log_file.write("=" * 100 + "\n")
        for result in all_results:
            log_file.write(
                f"{result['script_name']} | exit_code={result['return_code']} | duration={result['duration']}\n"
            )

    print(f"\nAll tasks finished. Log saved to:\n{log_filename}")


if __name__ == "__main__":
    main()
