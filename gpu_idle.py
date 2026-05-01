import argparse
import subprocess
import time

def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8'
    )
    res = []
    for i, line in enumerate(result.stdout.strip().split('\n')):
        used, total = line.split(',')
        res.append((int(used), int(total)))
    return res

def get_specific_gpu_mem(device_list = None):
    res = get_gpu_memory()

    _device_list = []
    if device_list is None:
        _device_list = range(len(res))
    else:
        for dd in device_list:
            for d in dd.split(','):
                _device_list.append(int(d))
    
    return [(idx, res[idx][0], res[idx][1]) for idx in _device_list]
    

def monitor(device_list, value, wait = 60, interval = 2):
    def check(res):
        for _, used, __ in res:
            if used >= value: return False
        return True
    
    wait_time = 0
    last_ch = 0
    while True:
        res = get_specific_gpu_mem(device_list)

        if check(res):
            # start to wait
            out = f'\rwaiting {wait_time}/{wait}s...'
            print(f"\r{''.join([' ' for _ in range(last_ch)])}",end='')
            print(out, end='')
            last_ch = len(out)
            time.sleep(interval)
            wait_time += interval
            if wait_time >= wait: break
        else:
            wait_time = 0
            line = [f'{idx}: {used}MB' for idx, used, _ in res]
            out = f'\rwaiting on [{" ".join(line)}] < {value}MB...'
            print(f"\r{''.join([' ' for _ in range(last_ch)])}",end='')
            print(out, end='')
            last_ch = len(out)
            time.sleep(interval)
    print('\nall gpu(s) are < value')

def detect(device_number, value, wait = 60, interval = 2):
    def check(res):
        ava = []
        for idx, used, __ in res:
            if used < value: ava.append(idx)
        return ava
    
    wait_time = 0
    last_ch = 0
    while True:
        res = get_specific_gpu_mem()
        ids = check(res)
        if len(ids) >= device_number:
            # start to wait
            out = f'\rwaiting {wait_time}/{wait}s...'
            print(f"\r{''.join([' ' for _ in range(last_ch)])}",end='')
            print(out, end='')
            last_ch = len(out)
            time.sleep(interval)
            wait_time += interval
            if wait_time >= wait: return ids[:device_number]
        else:
            wait_time = 0
            line = [f'{idx}: {used}MB' for idx, used, _ in res]
            out = f'\rwaiting on {device_number} cards in [{" ".join(line)}] < {value}MB...'
            print(f"\r{''.join([' ' for _ in range(last_ch)])}",end='')
            print(out, end='')
            last_ch = len(out)
            time.sleep(interval)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='gpu monitor')
    parser.add_argument("-M", "--monitor", help="monitor mode, device list as input, check if mem of device in device list < specific value (MB)", nargs='+', default=None)
    parser.add_argument("-D", "--detect", help="detect mode, number of device as input, return id of idle devices that mem < specific value (MB) when number of idle devices >= n", type=int, default=None)
    
    parser.add_argument("-m", "--memory", help="memory to monitor/detect. default=3000", type=int, default=3000)
    parser.add_argument("-i", "--interval", help="monitor/detector time interval (s). default=2", type=int, default=2)
    parser.add_argument("-w", "--wait", help="wait times of monitor's/detector's done (s). default=60", type=int, default=60)
    parser.add_argument("-o", "--output", help="file output name of detected idle device id(s). default=tdevice.txt", type=str, default='tdevice.txt')
    args = parser.parse_args()
    
    mem_value = args.memory
    monitor_mode = args.monitor
    detect_mode = args.detect
    if monitor_mode is None and detect_mode is None:
        res = get_specific_gpu_mem()
        for i, used, tot in res:
            print(f"GPU {i}: {used}MB / {tot}MB")
    elif monitor_mode is None and detect_mode is not None:
        ids = detect(detect_mode, mem_value, args.wait, args.interval)
        print('')
        out = ','.join([str(i) for i in ids])
        print(out)
        with open(args.output,'w',encoding='utf-8') as f:
            f.write(out)
    elif monitor_mode is not None and detect_mode is None:
        monitor(monitor_mode, mem_value, args.wait, args.interval)
    else:
        print("monitor and detect can't set simultaneously")
    
    
