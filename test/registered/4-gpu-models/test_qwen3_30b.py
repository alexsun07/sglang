import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=261, suite="stage-c-test-4-gpu-h100")
# AMD path uses the AITER attention backend (default on HIP). The CP code in
# aiter_backend.py is the same shape FA uses in flashattention_backend.py.
register_amd_ci(est_time=300, suite="stage-c-test-4-gpu-amd")

_IS_HIP = is_hip()


def _platform_extra_args():
    # AITER's custom all-reduce has a relaxed-memory-ordering data race that
    # surfaces when the entire decode forward is dispatched as a single cuda
    # graph (allreduce reads can land before peer writes). Falling back to
    # NCCL for the allreduce sidesteps it; cuda graph itself is fine.
    return ["--disable-custom-all-reduce"] if _IS_HIP else []


def _platform_extra_env():
    # When the attention CP process group exists, PyTorch's NCCL watchdog
    # races with cuda-graph capture on HIP: it polls hipEventQuery on Work
    # whose completion event lives on the capturing stream, which HIP
    # forbids (`hipErrorCapturedEvent`). Switching to blocking-wait mode
    # disables the watchdog thread and lets capture complete. Not needed
    # on CUDA — the FA backend's CP test runs without it.
    return {"TORCH_NCCL_BLOCKING_WAIT": "1"} if _IS_HIP else None

QWEN3_30B_MODEL_PATH = "Qwen/Qwen3-30B-A3B-FP8"

GSM8K_BASELINE_ACCURACY = 0.85


class TestQwen330B(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--moe-dp-size",
                "2",
                "--ep-size",
                "2",
                "--attn-cp-size",
                "2",
                "--enable-prefill-context-parallel",
                "--cuda-graph-max-bs",
                "32",
                "--max-running-requests",
                "32",
                "--trust-remote-code",
                "--disable-piecewise-cuda-graph",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
                *_platform_extra_args(),
            ],
            env=_platform_extra_env(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            model=self.model,
            eval_name="gsm8k",
            num_shots=5,
            num_examples=200,
            max_tokens=16000,
            num_threads=128,
            repeat=1,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], GSM8K_BASELINE_ACCURACY)


class TestQwen330BCP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--moe-dp-size",
                "1",
                "--ep-size",
                "4",
                "--attn-cp-size",
                "2",
                "--enable-prefill-context-parallel",
                "--cuda-graph-max-bs",
                "32",
                "--max-running-requests",
                "32",
                "--trust-remote-code",
                "--disable-piecewise-cuda-graph",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
                *_platform_extra_args(),
            ],
            env=_platform_extra_env(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            model=self.model,
            eval_name="gsm8k",
            num_shots=5,
            num_examples=200,
            max_tokens=16000,
            num_threads=128,
            repeat=1,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], GSM8K_BASELINE_ACCURACY)


if __name__ == "__main__":
    unittest.main()
