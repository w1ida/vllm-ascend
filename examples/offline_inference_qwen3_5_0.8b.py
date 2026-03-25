import os
os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/Ascend/nnal/atb/8.5.0/atb/cxx_abi_0/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from vllm import LLM, SamplingParams


def main():
    model_id = "Qwen/Qwen3.5-0.8B"
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.7,
        top_p=0.8,
        top_k=40,
        repetition_penalty=1.05,
    )

    print("Initializing LLM...")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
    )
    print("Generating outputs...")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Input: {output.prompt!r}")
        print(f"Output: {output.outputs[0].text!r}")

    print("Test completed successfully!")


if __name__ == "__main__":
    main()
