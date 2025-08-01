from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from vastai_sdk import VastAI
from pprint import pprint
from pint import UnitRegistry, Quantity

ureg = UnitRegistry()
# declare a new “currency” dimension:
ureg.define('currency = [currency]')
# now define some units and exchange rates (example rate: 1 EUR = 1.1 USD)
ureg.define('USD = 1 currency = dollar = $')

ureg.define('teraFLOP = [teraFLOP]')

@dataclass
class InstanceSpec:
    max_cost_per_hour: Quantity # The maximum rate per hour in USD
    num_gpus: int # The number of GPUs
    min_gpu_ram_size: Quantity # The minimum size per GPU RAM in GB
    min_gpu_ram_bw: Quantity # The minimum bandwidth of the GPU ram in GB/s
    min_total_tflops: Quantity # The minimum total TFLOP/s of the system
    min_downstream: Quantity # minimum download rate in MB/s
    min_upstream: Quantity # minimum upload rate in MB/s
    min_reliability: float # minimum reliability
    min_cuda_version: float = 12.6 # Minimum cuda version

    def to_query(self) -> str:
        return (
            f"num_gpus = {self.num_gpus} "
            f"gpu_ram >= {self.min_gpu_ram_size.to('GB').magnitude} "
            f"gpu_mem_bw >= {self.min_gpu_ram_bw.to('GB/s').magnitude} "
            f"total_flops >= {self.min_total_tflops.to('teraFLOP/s').magnitude} "
            f"inet_down >= {self.min_downstream.to('MB/s').magnitude} "
            f"inet_up >= {self.min_upstream.to('MB/s').magnitude} "
            f"cuda_vers >= {self.min_cuda_version} "
            f"reliability >= {self.min_reliability} "
            f"dph <= {self.max_cost_per_hour.to('USD').magnitude}"
        )

@dataclass
class InstanceOption:
    instance_id: int
    cost_per_hour: Quantity
    num_gpus: int
    gpu_name: str
    gpu_ram_size: Quantity
    gpu_ram_bw: Quantity
    total_tflops: Quantity
    downstream: Quantity
    upstream: Quantity
    reliability: float
    cuda_version: float

    @staticmethod
    def from_dict(dict: Dict[str, Any]) -> 'InstanceOption':
        return InstanceOption(
            instance_id=dict["id"],
            cost_per_hour=dict["dph_total_adj"] * ureg.USD,
            num_gpus=dict["num_gpus"],
            gpu_name=dict["gpu_name"],
            gpu_ram_size=dict["gpu_ram"] * ureg.MB,
            gpu_ram_bw=dict["gpu_mem_bw"] * ureg.GB / ureg.second,
            total_tflops=dict["total_flops"] * ureg.teraFLOP / ureg.second,
            downstream=dict["inet_down"] * ureg.MB / ureg.second,
            upstream=dict["inet_up"] * ureg.MB / ureg.second,
            reliability=dict["reliability"],
            cuda_version=dict["cuda_max_good"]
        )
    
    def pretty_str(self, indent: int = 4) -> str:
        indent_str = " " * indent
        
        return (
            f"{indent_str}Instance ID: {self.instance_id}\n"
            f"{indent_str}Cost per hour: {self.cost_per_hour.to('USD').magnitude:.2f} USD\n"
            f"{indent_str}Num GPUs: {self.num_gpus}\n"
            f"{indent_str}GPU Name: {self.gpu_name}\n"
            f"{indent_str}GPU RAM Size: {self.gpu_ram_size.to('GB').magnitude:.2f} GB\n"
            f"{indent_str}GPU RAM Bandwidth: {self.gpu_ram_bw.to('GB/s').magnitude:.2f} GB/s\n"
            f"{indent_str}Total TFLOPS: {self.total_tflops.to('teraFLOP/s').magnitude:.2f} TFLOP/s\n"
            f"{indent_str}Downstream: {self.downstream.to('MB/s').magnitude:.2f} MB/s\n"
            f"{indent_str}Upstream: {self.upstream.to('MB/s').magnitude:.2f} MB/s\n"
            f"{indent_str}Reliability: {self.reliability:.2f}\n"
            f"{indent_str}CUDA Version: {self.cuda_version}\n"
        )


class InstanceKind(Enum):
    Dev = InstanceSpec(
        max_cost_per_hour=0.5 * ureg.USD,
        num_gpus=1,
        min_reliability=0.9,
        min_gpu_ram_size=15 * ureg.GB,
        min_gpu_ram_bw=700 * ureg.GB / ureg.second,
        min_total_tflops=10 * ureg.teraFLOP / ureg.second,
        min_downstream=1000.0 * ureg.MB / ureg.second,
        min_upstream=1000.0 * ureg.MB / ureg.second
    )
    Train = InstanceSpec(
        max_cost_per_hour=4 * ureg.USD,
        num_gpus=1,
        min_reliability=0.95,
        min_gpu_ram_size=40 * ureg.GB,
        min_gpu_ram_bw=3000 * ureg.GB / ureg.second,
        min_total_tflops=30 * ureg.teraFLOP / ureg.second,
        min_downstream=1000 * ureg.MB / ureg.second,
        min_upstream=1000 * ureg.MB / ureg.second
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


if __name__ == "__main__":
    api = VastAI()
    instance_option = find_instance(api=api, instance_kind=InstanceKind.Dev)
    
    if instance_option is None:
        raise RuntimeError("Could not find an instance for the desired spec")

    # TODO: create the instance
    # TODO: automatically run all setup scripts on the instance
