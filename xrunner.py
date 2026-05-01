import os
import yaml
import logging
import socket
import argparse
import subprocess
from copy import deepcopy

from dataclasses import dataclass, field, asdict
from gpu_idle import detect
from typing import Union, List, Any, Dict, Literal

help = """
=====================================================================================================
Here is the config template for running a python script through x runner, * means required.

```yml

config:
    run: # str: .py file path *
    step-run: # str: run a command after every step with env vars
    pre-run: # str: run a command before the main program.
    post-run: # str: run a command after the main program.
    err-run: # str: run a command when error occupied.

    uv: # bool: run with uv. default=False
    accelerator: # bool: run with accelerator. default=False
    accelerator-port: # int: specific port of accelerator, set None to auto detect. default=None
    auto-idle-gpu: 
        count: # int: number of gpu in need, None means disabled the auto gpu func. default=None
        max-memory-detect: # int: before run script, program will sleep until gpu mem small than this value(MB). default=3000
        wait: # int: when memory size valid, the program will wait this value(sec). default=60
        wait-every-time: # bool: wait before every single run. default=False
        interval: # int: interval of gpu memory detect. default=2
    log: # str: a log file that indicate what param running currently, set same will log with the same of config name, set None to disable it. default=same
    log-cmd: # bool: log cmd that run. default=true
    log-var: # list[str]: log specific vars. default=[]

param: # your own params here, environment vars begin with $. And other will trades with prog var, it will input to python script begin with prefix '--'
    - p1: ...
    - p2: [...] # support arrarys
    - p3: ...
    ...


pairs: # not implement, link two or more parameters, these parameters will synchronized by there idxs when input to program, and they must have the same array length
    env:
        - [p1, p2, ..., pn]
        - [q1, q2, ..., qn]
        ...
    prog:
        ...
    
```
=====================================================================================================
"""

@dataclass
class AutoIdleGPU:
    count: int = None
    maxMemoryDetect: int = 3000
    wait: int = 60
    waitEveryTime: bool = False
    interval: int = 2

@dataclass
class Config:
    run: str
    stepRun: str = ""
    preRun: str = ""
    postRun: str = ""
    errRun: str = ""
    uv: bool = False
    accelerator: bool = False
    acceleratorPort: int = None
    autoIdleGPU: AutoIdleGPU = field(default_factory=AutoIdleGPU)
    log: str = "same"
    logCmd: bool = True
    logVar: list[str] = field(default_factory=list())

class RunnerTask:
    def __init__(self, config_file):
        self.round = 0
        self.stop = False
        self.config = RunnerTask.read_config(config_file["config"])
        if "param" in config_file:
            param = RunnerTask.normalize_param(config_file["param"])
        else:
            param = {}
        self.param: dict[Union[str, List[str]], Dict[Literal["val",  "type"], Union[List[Union[Any, List[Any]]],  str]]] = param # 对于多组绑定的param可以采用val=List[name]: List[val]来表示
        tot_params = 1
        for pname in param:
            tot_params *= len(param[pname]["val"])
        self.tot_params = tot_params

        if self.config.log:
            if self.config.log == "same":
                self.config.log = os.path.splitext(args.file)[0] + ".log"
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename=self.config.log)
        else:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    @staticmethod
    def read_config(config_dict: dict):
        name_map = {
            "pre-run": "preRun",
            "post-run": "postRun",
            "step-run": "stepRun",
            "err-run": "errRun",
            "log-cmd": "logCmd",
            "log-var": "logVar",
            "accelerator-port": "acceleratorPort",
            "max-memory-detect": "maxMemoryDetect",
            "wait-every-time": "waitEveryTime"
        }

        args = {}
        required = ["run"]
        for r in required:
            if r not in config_dict.keys():
                raise NameError(f"`{r}` must be presented in `config`!")
        for name in config_dict:
            if name != "auto-idle-gpu":
                _name = name_map[name] if name in name_map else name
                args[_name] = config_dict[name]
            else:
                gpu_args = {}
                for gname in config_dict["auto-idle-gpu"]:
                    _name = name_map[gname] if gname in name_map else gname
                    gpu_args[_name] = config_dict["auto-idle-gpu"][gname]
                args["autoIdleGPU"] = AutoIdleGPU(**gpu_args)
        return Config(**args)
    
    @staticmethod
    def normalize_param(param):
        # TODO: implement the linked params

        p = {}
        for name in param:
            val = param[name]
            p[name] = {
                "val": val if type(val) == list else [val],
                "type": "env" if name[0] == '$' else "prog"
            }
        return p

    def search(self):
        parray = [(n, self.param[n]) for n in self.param.keys()]
        cur = []
        
        def dfs(level):
            if self.stop: return
            if level >= len(parray):
                params = {data["name"]: { "val": data["val"], "type": data["type"] } for data in cur}
                try:
                    self.run(params)
                except Exception as e:
                    if type(e) == KeyboardInterrupt:
                        self.stop = True
                    else:
                        logging.exception("Something EQrror:")
                self.round += 1
                logging.info(f"{self.round * 100 / self.tot_params:.2f}% [{self.round}/{self.tot_params}]")
                return
            for val in parray[level][1]["val"]:
                cur.append({
                    "name": parray[level][0],
                    "val": val,
                    "type": parray[level][1]["type"]
                }) # name=union[param_name,list[param_name]], val=union[val,list[val]]
                dfs(level + 1)
                cur.pop()

        logging.info("======================================== START PARAM SEARCH ========================================")
        logging.info(f"The prog is {self.config.run}")
        dfs(0)
        # run post run
        logging.info(f"Start post-run cmd: {self.config.postRun}")
        subprocess.run(self.config.postRun, shell=True)
        logging.info("========================================= END PARAM SEARCH =========================================")

    @staticmethod
    def find_free_port(start_port, end_port, host='127.0.0.1'):
        for port in range(start_port, end_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                    # 绑定成功，说明端口未被占用
                    return port
                except OSError:
                    # 绑定失败，端口可能被占用
                    continue
        return None

    def run(self, params: dict):
        # TODO: implement the linked params
        
        base_env = os.environ.copy()

        # 1. auto gpu before run
        if self.config.autoIdleGPU.count is not None:
            wait = self.config.autoIdleGPU.wait if self.config.autoIdleGPU.waitEveryTime or self.round == 0 else 0
            device = detect(self.config.autoIdleGPU.count, self.config.autoIdleGPU.maxMemoryDetect, wait, self.config.autoIdleGPU.interval)
            print()
            device = ",".join([str(i) for i in device])
            logging.info(f"Selected devices: {device}")
        
        # 2. pre-run cmd
        if self.round == 0:
            logging.info(f"Start pre-run cmd: {self.config.preRun}")
            subprocess.run(self.config.preRun, shell=True, env={"device": device, **base_env})
            
            
        env_cmd = ""
        # 3. set env
        env_cmd += f'export CUDA_VISIBLE_DEVICES={device}\n'

        for pname in params:
            if params[pname]["type"] == "env":
                env_cmd += f'export {pname[1:]}={params[pname]["val"]}\n'

        # 4. generate main cmd
        if self.config.uv:
            main_cmd = "uv run "
        else:
            main_cmd = ""
        if self.config.accelerator:
            main_cmd += "accelerate launch"
            if self.config.acceleratorPort:
                port = self.config.acceleratorPort
            else:
                port = RunnerTask.find_free_port(29500,29599) # FIXME: 也许还有点bug
                if port is None:
                    raise FileNotFoundError("No port free...")
            main_cmd += f" \\\n    --main_process_port {port}"
        else:
            main_cmd += "\npython"
        main_cmd += f" \\\n    {self.config.run}"

        # 5. generate main param
        for pname in params:
            if params[pname]["type"] == "env": continue
            main_cmd += f" \\\n        --{pname} {params[pname]['val']}"
        
        # 6. Runnnn!!!!
        plog = "=== Running with: ==="
        for pname in self.config.logVar:
            if pname in params:
                plog += f"\n{pname}: {params[pname]['val']}"
        if plog != "=== Running with: ===": logging.info(plog)

        main_run_log = f"# set env vars\n{env_cmd}\n# run cmd\n{main_cmd}"
        if self.config.logCmd: logging.info(f"Running...\n{main_run_log}")

        subprocess.run(f"{env_cmd}\n{main_cmd}", shell=True)

        # 7. step-run
        logging.info(f"Start step-run cmd: {self.config.stepRun}")
        subprocess.run(f"{env_cmd}\n{self.config.stepRun}", shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'X Runner, a simple python runner for parameters search.', epilog=help, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("file", help="config file path")
    parser.add_argument("-o", help="output the simple .sh for config, not suport for linked params.") # TODO: not implement
    args = parser.parse_args()
    with open(args.file, 'r', encoding='utf-8') as f:
        config_file = yaml.safe_load(f)

    if "config" not in config_file:
        raise NameError("`config` must be presented in yml!")
    
    run = RunnerTask(config_file)
    run.search()