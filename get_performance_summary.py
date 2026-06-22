import re

def parse_time(time_str):
    minutes_str, seconds_str = time_str.replace('s', '').split('m ')
    total_seconds = int(minutes_str) * 60 + int(seconds_str)
    return total_seconds

def parse_peak_vram(peak_mem_str):
    vram = float(peak_mem_str.replace(' GB', ''))
    return vram

def parse_line(match):
    epoch = int(match.group(1))
    time = parse_time(match.group(2).strip())
    peak_mem = parse_peak_vram(match.group(3).strip())

    return {
            "Epoch": epoch,
            "Time": time,
            "Peak VRAM": peak_mem
           }

def get_metrics(file):
    metrics = []
    with open(file, 'r') as file:
        for line in file:
            match = re.search(pattern, line)

            if match:
                metrics.append(parse_line(match))

    num_epochs = len(metrics)
    avg_time_seconds = sum(row['Time'] for row in metrics) / num_epochs
    avg_vram = sum(row['Peak VRAM'] for row in metrics) / num_epochs

    return avg_time_seconds, avg_vram

log_files = [
        'logs/SPEAR_CNN_EMULATOR-1.15423957',
        'logs/SPEAR_CNN_EMULATOR-2.15423958',
        'logs/SPEAR_UNET_EMULATOR-1.15423959',
        'logs/SPEAR_UNET_EMULATOR-2.15423960'
        ]

pattern = r"\[Epoch (\d+)\] Time: ([^\|]+)\s*\|.*?Peak VRAM: ([\d.]+ GB)"

for file in log_files:
    print(get_metrics(file))
