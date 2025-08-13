from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from vastai_sdk import VastAI
from pprint import pprint
from pint import UnitRegistry, Quantity
from pathlib import Path, PurePath
import asyncio
from time import sleep
from ssh_setup import SSHSession
from github_config import get_github_api
from github import Github, AuthenticatedUser

ureg = UnitRegistry()
# declare a new “currency” dimension:
ureg.define('currency = [currency]')
# now define some units and exchange rates (example rate: 1 EUR = 1.1 USD)
ureg.define('USD = 1 currency = dollar = $')

ureg.define('teraFLOP = [teraFLOP]')

@dataclass
class InstanceSpec:
    max_cost_per_hour: Quantity # The maximum rate per hour in USD
    min_cpu_ram_size: Quantity # The minimum CPU ram
    num_gpus: int # The number of GPUs
    min_gpu_ram_size: Quantity # The minimum size per GPU RAM in GB
    min_gpu_ram_bw: Quantity # The minimum bandwidth of the GPU ram in GB/s
    min_total_tflops: Quantity # The minimum total TFLOP/s of the system
    min_downstream: Quantity # minimum download rate in MB/s
    min_upstream: Quantity # minimum upload rate in MB/s
    min_disk_space: Quantity # minimum disk spaces
    min_reliability: float # minimum reliability
    min_cuda_version: float = 12.6 # Minimum cuda version

    def to_query(self) -> str:
        return (
            f"num_gpus = {self.num_gpus} "
            f"cpu_ram >= {self.min_cpu_ram_size.to('GB').magnitude} "
            f"gpu_ram >= {self.min_gpu_ram_size.to('GB').magnitude} "
            f"gpu_mem_bw >= {self.min_gpu_ram_bw.to('GB/s').magnitude} "
            f"total_flops >= {self.min_total_tflops.to('teraFLOP/s').magnitude} "
            f"inet_down >= {self.min_downstream.to('MB/s').magnitude} "
            f"inet_up >= {self.min_upstream.to('MB/s').magnitude} "
            f"cuda_vers >= {self.min_cuda_version} "
            f"reliability >= {self.min_reliability} "
            f"dph <= {self.max_cost_per_hour.to('USD').magnitude} "
            f"disk_space >= {self.min_disk_space.to('GB').magnitude}"
        )

@dataclass
class InstanceOption:
    instance_id: int
    cost_per_hour: Quantity
    cpu_ram: Quantity
    num_gpus: int
    gpu_name: str
    gpu_ram_size: Quantity
    gpu_ram_bw: Quantity
    total_tflops: Quantity
    downstream: Quantity
    upstream: Quantity
    reliability: float
    cuda_version: float
    available_disk_space: Quantity

    @staticmethod
    def from_dict(dict: Dict[str, Any]) -> 'InstanceOption':
        return InstanceOption(
            instance_id=dict["id"],
            cost_per_hour=dict.get("dph_total_adj", dict.get("dph_total")) * ureg.USD,
            cpu_ram=dict["cpu_ram"] * ureg.MB,
            num_gpus=dict["num_gpus"],
            gpu_name=dict["gpu_name"],
            gpu_ram_size=dict["gpu_ram"] * ureg.MB,
            gpu_ram_bw=dict["gpu_mem_bw"] * ureg.GB / ureg.second,
            total_tflops=dict["total_flops"] * ureg.teraFLOP / ureg.second,
            downstream=dict["inet_down"] * ureg.MB / ureg.second,
            upstream=dict["inet_up"] * ureg.MB / ureg.second,
            reliability=dict.get("reliability", dict.get("reliability2", 0.0)),
            cuda_version=dict["cuda_max_good"],
            available_disk_space=dict["disk_space"] * ureg.GB
        )
    
    def pretty_str(self, indent: int = 4) -> str:
        indent_str = " " * indent
        
        return (
            f"{indent_str}Instance ID: {self.instance_id}\n"
            f"{indent_str}Cost per hour: {self.cost_per_hour.to('USD').magnitude:.2f} USD\n"
            f"{indent_str}CPU RAM: {self.cpu_ram.to('GB').magnitude:.2f} GB\n"
            f"{indent_str}Num GPUs: {self.num_gpus}\n"
            f"{indent_str}GPU Name: {self.gpu_name}\n"
            f"{indent_str}GPU RAM Size: {self.gpu_ram_size.to('GB').magnitude:.2f} GB\n"
            f"{indent_str}GPU RAM Bandwidth: {self.gpu_ram_bw.to('GB/s').magnitude:.2f} GB/s\n"
            f"{indent_str}Total TFLOPS: {self.total_tflops.to('teraFLOP/s').magnitude:.2f} TFLOP/s\n"
            f"{indent_str}Downstream: {self.downstream.to('MB/s').magnitude:.2f} MB/s\n"
            f"{indent_str}Upstream: {self.upstream.to('MB/s').magnitude:.2f} MB/s\n"
            f"{indent_str}Reliability: {self.reliability:.2f}\n"
            f"{indent_str}CUDA Version: {self.cuda_version}\n"
            f"{indent_str}Disk Space: {self.available_disk_space:.2f}\n"
        )
    
    def fulfills(self, instance_spec: InstanceSpec) -> bool:
        """
        Check if an instance option is within spec
        """
        return (
            self.num_gpus==instance_spec.num_gpus and
            self.cost_per_hour<=instance_spec.max_cost_per_hour and
            self.gpu_ram_size>=instance_spec.min_gpu_ram_size and
            self.gpu_ram_bw>=instance_spec.min_gpu_ram_bw and
            self.total_tflops>=instance_spec.min_total_tflops and
            self.upstream>=instance_spec.min_upstream and
            self.downstream>=instance_spec.min_downstream and
            self.cuda_version>=instance_spec.min_cuda_version and
            self.reliability>=instance_spec.min_reliability
        )


@dataclass
class RunningInstanceOption(InstanceOption):
    ip_addr: str
    ssh_host: str
    ssh_port: int
    static_ip: bool
    
    @staticmethod
    def from_dict(dict: Dict[str, Any]) -> 'RunningInstanceOption':
        instance_opt = InstanceOption.from_dict(dict)
        
        return RunningInstanceOption(
            **instance_opt.__dict__,
            ip_addr=dict["public_ipaddr"],
            ssh_host=dict["ssh_host"],
            ssh_port=dict["ssh_port"]+1,
            static_ip=dict["static_ip"]
        )


class InstanceKind(Enum):
    Dev = InstanceSpec(
        max_cost_per_hour=0.5 * ureg.USD,
        min_cpu_ram_size=64 * ureg.GB,
        num_gpus=1,
        min_reliability=0.9,
        min_gpu_ram_size=15 * ureg.GB,
        min_gpu_ram_bw=700 * ureg.GB / ureg.second,
        min_total_tflops=10 * ureg.teraFLOP / ureg.second,
        min_downstream=2000.0 * ureg.MB / ureg.second,
        min_upstream=2000.0 * ureg.MB / ureg.second,
        min_disk_space=200 * ureg.GB
    )
    Train = InstanceSpec(
        max_cost_per_hour=4 * ureg.USD,
        min_cpu_ram_size=128 * ureg.GB,
        num_gpus=1,
        min_reliability=0.95,
        min_gpu_ram_size=40 * ureg.GB,
        min_gpu_ram_bw=3000 * ureg.GB / ureg.second,
        min_total_tflops=30 * ureg.teraFLOP / ureg.second,
        min_downstream=1000 * ureg.MB / ureg.second,
        min_upstream=1000 * ureg.MB / ureg.second,
        min_disk_space=200 * ureg.GB
    )


def find_instance(api: VastAI, instance_kind: InstanceKind) -> Optional[InstanceOption]:
    """
    Looks for an instance fitting for the specified purpose and returns the ID
    """
    
    query_str = instance_kind.value.to_query()
    
    result: List[Dict[str, Any]] = api.search_offers(
        type="on-demand",
        query=query_str,
        order="dph"
    ) # type: ignore
    
    structured_results = [InstanceOption.from_dict(item) for item in result]
    
    if len(structured_results) == 0:
        return None
    
    chosen_instance = structured_results[0]
    
    print(f"**Chosen instance for kind '{instance_kind.name}'**\n{chosen_instance.pretty_str()}")
    
    # use the cheapest one that fulfills the specs
    return chosen_instance


def create_instance(api: VastAI, template_id: int, instance_id: int):
    pass


def create_new_instance_from_spec(api: VastAI, instance_kind: InstanceKind) -> int:
    instance_option = find_instance(api=api, instance_kind=instance_kind)
    
    if instance_option is None:
        raise RuntimeError("Could not find an instance for the desired spec")

    template_hash = "aca0a2979e975f993dae3e127d083913" # my template ...

    create_result = api.create_instance(
        id=instance_option.instance_id,
        template_hash=template_hash,
        ssh=True,
        disk=instance_kind.value.min_disk_space.to('GB').magnitude
    )
    
    return instance_option.instance_id


def show_existing_instances_for_user(api: VastAI) -> List[RunningInstanceOption]:
    results: List[Dict[str, Any]] = api.show_instances() # type: ignore
    
    parsed_results = []
    
    for instance in results:
        if instance["actual_status"] not in ["running"]:
            continue
        parsed_results.append(RunningInstanceOption.from_dict(instance))
    
    return parsed_results

def check_existing_instances(api: VastAI, instance_spec: InstanceSpec) -> Optional[RunningInstanceOption]:
    """
    Checks if a currently running instance fulfills the spec
    """
    running_instances = show_existing_instances_for_user(api=api)
    
    for running_instance in running_instances:
        if running_instance.fulfills(instance_spec):
            return running_instance
    return None

def fetch_ssh_key(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def wait_for_instance(api: VastAI, instance_spec: InstanceSpec) -> RunningInstanceOption:
    """
    Wait until the instance is ready
    """
    instance_with_spec = check_existing_instances(api=api, instance_spec=instance_spec)
    
    while instance_with_spec is None:
        print("Instance is not yet ready, waiting 10 seconds")
        sleep(10)
        instance_with_spec = check_existing_instances(api=api, instance_spec=instance_spec) # type: ignore
    print("Instance is ready")
    
    return instance_with_spec

def handle_setup(api: VastAI, instance_kind: InstanceKind) -> RunningInstanceOption:
    running_instance = check_existing_instances(api=api, instance_spec=instance_kind.value)
    
    if running_instance is None:
        new_instance_id = create_new_instance_from_spec(api=api, instance_kind=instance_kind)
        
        running_instance = wait_for_instance(api=api, instance_spec=instance_kind.value)
    
    return running_instance


async def setup_machine(api: VastAI, github: Github, instance_info: RunningInstanceOption):
    async with SSHSession(instance_info.ssh_host, port=instance_info.ssh_port,
                          username='root', client_keys=['~/.ssh/id_rsa']) as sess:
        # get python version
        result = await sess.run('/venv/main/bin/python -V')
        print(f"python version: {result}")
        
        # Nvim
        print("Installing nvim")
        result = await sess.run('sudo apt install neovim python3-neovim -y')
        print(f"result: {result}")
        
        #PyTorch
        print("Installing PyTorch")
        await sess.run('PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126')
        result = await sess.run("/venv/main/bin/python -m pip install torch==2.7.1 torchvision torchaudio --index-url $PYTORCH_INDEX_URL")
        print(f"result: {result}")
        
        # PyTorch helpers
        print("Installing additional stuff")
        result = await sess.run("/venv/main/bin/python -m pip install packaging ninja wheel setuptools setuptools-scm")
        print(f"result: {result}")
        
        # Flash Attn
        print("Installing Flash attention")
        result = await sess.run("MAX_JOBS=8 /venv/main/bin/python -m pip install flash-attn==2.8.2 --no-build-isolation")
        print(f"result: {result}")
        
        # ECDSA keys
        print("Generating ECDSA keys")
        result = await sess.run("ssh-keygen -t ecdsa -b 521 -C \"hendrik.leier75@gmail.com\" -f ~/.ssh/id_rsa -N \"\" -q </dev/null")
        print(f"result: {result}")
        key = await sess.run("cat ~/.ssh/id_rsa.pub")
        
        print(f"key for git: {key}")
        
        # Add key to github
        print("adding key to github")
        github_user = github.get_user()
        github_user.create_key(title=f"VastAI box {instance_info.instance_id}", key=key) # type: ignore
        print("Key added")
        
        # TODO: Need to add github key to known hosts manually
        # Clone repo
        print("Cloning repo")
        result = await sess.run("git clone git@github.com:HendrikLeier/HRM.git")
        print(f"result: {result}")
        
        print("Install requirements")
        await sess.run("cd HRM")
        result = await sess.run("/venv/main/bin/python -m pip install -r requirements.txt")
        print(f"Result: {result}")

if __name__ == "__main__":
    api = VastAI()
    github = get_github_api()
    
    running_instance = handle_setup(api=api, instance_kind=InstanceKind.Dev)
    
    local_ssh_key = fetch_ssh_key(Path("/Users/hendrikleier/.ssh/id_rsa.pub"))
        
    result = api.attach_ssh(instance_id=running_instance.instance_id, ssh_key=local_ssh_key)

    asyncio.run(setup_machine(api=api, github=github, instance_info=running_instance))

    print()
    # TODO: create the instance
    # TODO: automatically run all setup scripts on the instance
