@dataclass
class FullyShardedDataParallelPlugin:
    """
    This plugin is used to enable fully sharded data parallelism.
    """

    sharding_strategy: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP Sharding Strategy of type `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy`"
        },
    )
    backward_prefetch: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP Backward Prefetch of type `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch`"
        },
    )
    mixed_precision_policy: "typing.Any" = field(
        default=None,
        metadata={
            "help": "A config to enable mixed precision training with FullyShardedDataParallel. "
            "The 3 flags that are set are `param_dtype`, `reduce_dtype`, `buffer_dtype`. "
            "Each flag expects `torch.dtype` as the value. "
            "It is of type `torch.distributed.fsdp.fully_sharded_data_parallel.MixedPrecision`."
        },
    )
    auto_wrap_policy: Optional[Callable] = field(
        default=None,
        metadata={"help": "A callable specifying a policy to recursively wrap layers with FSDP"},
    )
    cpu_offload: "typing.Any" = field(
        default=None,
        metadata={
            "help": "Decides Whether to offload parameters and gradients to CPU. "
            "It is of type `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffload`."
        },
    )
    ignored_modules: Optional[Iterable[torch.nn.Module]] = field(
        default=None,
        metadata={"help": "A list of modules to ignore for FSDP."},
    )
    state_dict_type: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP State Dict Type of type `torch.distributed.fsdp.fully_sharded_data_parallel.StateDictType`"
        },
    )
    state_dict_config: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP State Dict Config of type `torch.distributed.fsdp.fully_sharded_data_parallel.StateDictConfig`"
        },
    )
    optim_state_dict_config: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP Optimizer State Dict Config of type `torch.distributed.fsdp.fully_sharded_data_parallel.OptimStateDictConfig`"
        },
    )
    limit_all_gathers: bool = field(
        default=False,
        metadata={
            "help": "If False, then FSDP allows the CPU thread to schedule all-gathers "
            "without any extra synchronization. If True, then FSDP explicitly synchronizes the CPU thread to prevent "
            "too many in-flight all-gathers. This bool only affects the sharded strategies that schedule all-gathers. "
            "Enabling this can help lower the number of CUDA malloc retries."
        },
    )
    use_orig_params: bool = field(
        default=True,
        metadata={
            "help": "If True, allows non-uniform `requires_grad` during init, which means support for interspersed frozen and trainable paramteres. "
            "Useful in cases such as parameter-efficient fine-tuning. "
            "Please refer this [blog](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019)"
        },
    )
    param_init_fn: Optional[Callable[[torch.nn.Module], None]] = field(
        default=None,
        metadata={
            "help": "A Callable[torch.nn.Module] -> None that specifies how modules "
            "that are currently on the meta device should be initialized onto an actual device."
        },
    )
    sync_module_states: bool = field(
        default=True,
        metadata={
            "help": "If True, each individually wrapped FSDP unit will broadcast module parameters from rank 0 "
            "to ensure they are the same across all ranks after initialization"
        },
    )
    forward_prefetch: bool = field(
        default=False,
        metadata={
            "help": "If True, then FSDP explicitly prefetches the next upcoming "
            "all-gather while executing in the forward pass. only use with Static graphs."
        },
    )
    activation_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, activation checkpointing is a technique to reduce memory usage by clearing activations of "
            "certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time "
            "for reduced memory usage."
        },
    )

    def __post_init__(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch, CPUOffload, ShardingStrategy

        prefix = "FSDP_"
        if self.sharding_strategy is None:
            self.sharding_strategy = ShardingStrategy(int(os.environ.get(prefix + "SHARDING_STRATEGY", 1)))

        if self.cpu_offload is None:
            if str_to_bool(os.environ.get(prefix + "OFFLOAD_PARAMS", "False")) == 1:
                self.cpu_offload = CPUOffload(offload_params=True)
            else:
                self.cpu_offload = CPUOffload(offload_params=False)

        if self.backward_prefetch is None:
            prefetch_policy = os.environ.get(prefix + "BACKWARD_PREFETCH", "NO_PREFETCH")
            if prefetch_policy != FSDP_BACKWARD_PREFETCH[-1]:
                self.backward_prefetch = BackwardPrefetch(FSDP_BACKWARD_PREFETCH.index(prefetch_policy) + 1)

        if self.state_dict_type is None:
            state_dict_type_policy = os.environ.get(prefix + "STATE_DICT_TYPE", "FULL_STATE_DICT")
            self.set_state_dict_type(state_dict_type_policy)
        self.use_orig_params = str_to_bool(os.environ.get(prefix + "USE_ORIG_PARAMS", "False")) == 1
        self.sync_module_states = str_to_bool(os.environ.get(prefix + "SYNC_MODULE_STATES", "True")) == 1
        self.forward_prefetch = str_to_bool(os.environ.get(prefix + "FORWARD_PREFETCH", "False")) == 1
        self.activation_checkpointing = str_to_bool(os.environ.get(prefix + "ACTIVATION_CHECKPOINTING", "False")) == 1

        if self.sync_module_states:
            if is_npu_available():
                device = torch.npu.current_device()
            elif is_cuda_available():
                device = torch.cuda.current_device()
            elif is_xpu_available():
                device = torch.xpu.current_device()
            else:
                raise RuntimeError(
                    "There are currently no available devices found, must be one of 'XPU', 'CUDA', or 'NPU'."
                )
            self.param_init_fn = lambda x: x.to_empty(device=device, recurse=False)

    @staticmethod
    def get_module_class_from_name(module, name):
        """
        Gets a class from a module by its name.

        Args:
            module (`torch.nn.Module`): The module to get the class from.
            name (`str`): The name of the class.
        """
        modules_children = list(module.children())
        if module.__class__.__name__ == name:
            return module.__class__
        elif len(modules_children) == 0:
            return
        else:
            for child_module in modules_children:
                module_class = FullyShardedDataParallelPlugin.get_module_class_from_name(child_module, name)
                if module_class is not None:
                    return module_class

    def set_auto_wrap_policy(self, model):
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

        default_transformer_cls_names_to_wrap = (
            ",".join(model._no_split_modules) if getattr(model, "_no_split_modules", None) is not None else ""
        )
        if self.auto_wrap_policy is None:
            auto_wrap_policy = os.environ.get("FSDP_AUTO_WRAP_POLICY", "NO_WRAP")
            if auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[0]:
                transformer_cls_names_to_wrap = os.environ.get(
                    "FSDP_TRANSFORMER_CLS_TO_WRAP", default_transformer_cls_names_to_wrap
                ).split(",")
                transformer_cls_to_wrap = set()
                for layer_class in transformer_cls_names_to_wrap:
                    transformer_cls = FullyShardedDataParallelPlugin.get_module_class_from_name(model, layer_class)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        transformer_cls_to_wrap.add(transformer_cls)

                self.auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls=transformer_cls_to_wrap,
                )
            elif auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[1]:
                min_num_params = int(os.environ.get("FSDP_MIN_NUM_PARAMS", 0))
                if min_num_params > 0:
                    self.auto_wrap_policy = functools.partial(
                        size_based_auto_wrap_policy, min_num_params=min_num_params
                    )

    def set_mixed_precision(self, mixed_precision):
        if mixed_precision == "fp16":
            dtype = torch.float16
        elif mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown mixed precision value: {mixed_precision}")
        from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision

        if self.mixed_precision_policy is None:
            self.mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    def set_state_dict_type(self, state_dict_type_policy):
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            StateDictType,
        )

        self.state_dict_type = StateDictType(FSDP_STATE_DICT_TYPE.index(state_dict_type_policy) + 1)

        if self.state_dict_type == StateDictType.FULL_STATE_DICT:
            if self.state_dict_config is None:
                self.state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            if self.optim_state_dict_config is None:
                self.optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
