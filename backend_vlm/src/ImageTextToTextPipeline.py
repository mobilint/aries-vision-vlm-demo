import torch
import logging
import traceback
import re
from threading import Event, Thread
from typing import Callable, Dict, Optional
from contextlib import contextmanager
import types, functools, inspect
from qbruntime import Accelerator

from transformers import TextIteratorStreamer, GenerationConfig, AutoProcessor, AutoModelForImageTextToText


@contextmanager
def get_image_features_callback(model, callback: Optional[Callable] = None):
    original = model.get_image_features

    @functools.wraps(original)
    def patched(*args, **kwargs):
        out = original(*args, **kwargs)
        if callback:
            callback()
        return out

    model.get_image_features = types.MethodType(patched, model)
    try:
        model.get_image_features.__signature__ = inspect.signature(original)
    except Exception:
        pass
    try:
        yield
    finally:
        model.get_image_features = original


class StopOnSignalTextIteratorStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, stop_event, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.stop_event = stop_event

    def put(self, value):
        if self.stop_event.is_set():    
            self.end_of_stream = True
            raise StopIteration()
        super().put(value)


class ImageTextToTextPipeline:
    def __init__(self):
        self._configure_logging()
        self.model_id = self._select_device_and_model()
        self.model, self.processor = self._load_model_and_processor(self.model_id)
        self.sessions: Dict[str, Dict] = {}

    def _configure_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    def _select_device_and_model(self) -> str:
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        self.original_model_id = model_name

        gpu_available = torch.cuda.is_available()
        npu_available = False
        
        try:
            acc = Accelerator()
            del acc
            npu_available = True
        except:
            pass

        logging.info(f'[DEVICE] GPU: {"O" if gpu_available else "X"}, NPU: {"O" if npu_available else "X"}')
        
        if gpu_available == False and npu_available == False:
            raise SystemError("No AI Accelerator Found!")
        
        self.is_npu = npu_available

        if npu_available:
            return re.sub(r"^[^/]+", "mobilint", model_name)
        if gpu_available:
            return model_name

        raise RuntimeError("[DEVICE] No available AI accelerator!")

    def _load_model_and_processor(self, model_id: str):
        logging.info(f"Loading processor for model: {model_id}")
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True
        )

        logging.info(f"Loading model: {model_id}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to("cpu" if self.is_npu else "cuda:0")
        return model, processor

    def reset_session(self, session_id: str):
        existing_session = self.sessions.get(session_id, {})
        self.sessions[session_id] = {
            "past_key_values": None,
            "system_prompt": existing_session.get("system_prompt", ""),
            "inter_prompt": existing_session.get("inter_prompt", ""),
            "conversation": [],
        }
        self._apply_session_prompt(session_id)
        logging.info(f"[{session_id}] - Cache has been reset.")

    def set_session_prompts(self, session_id: str, system_prompt: str, inter_prompt: str = ""):
        if session_id not in self.sessions:
            self.reset_session(session_id)

        self.sessions[session_id]["system_prompt"] = system_prompt or ""
        self.sessions[session_id]["inter_prompt"] = inter_prompt or ""
        self._apply_session_prompt(session_id)
        logging.info(f"[{session_id}] - Session prompts updated.")

    def _apply_session_prompt(self, session_id: str):
        session = self.sessions[session_id]
        prompt_parts = [session.get("system_prompt", "").strip()]
        inter_prompt = session.get("inter_prompt", "").strip()

        if inter_prompt:
            prompt_parts.append(inter_prompt)

        merged_prompt = "\n\n".join(part for part in prompt_parts if part)

        session["past_key_values"] = None
        session["conversation"] = []

        if merged_prompt:
            session["conversation"].append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": merged_prompt,
                        }
                    ],
                }
            )

    def generate_stream(
        self,
        session_id: str,
        image_url: Optional[str],
        text_prompt: str,
        on_token: Callable,
        on_end: Callable,
        on_image_processing_done: Optional[Callable] = None,
    ):
        if self.model is None or self.processor is None:
            raise RuntimeError("Pipeline is not available.")

        if (
            session_id in self.sessions
            and self.sessions[session_id].get("task_thread")
            and self.sessions[session_id]["task_thread"].is_alive()
        ):
            logging.warning(f"[{session_id}] - Generation is already in progress")
            return

        if session_id not in self.sessions:
            self.reset_session(session_id)

        stop_event = Event()
        streamer = StopOnSignalTextIteratorStreamer(
            self.processor.tokenizer,
            stop_event,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def task():
            try:
                content = []
                if image_url:
                    content.append({"type": "image", "url": image_url})
                content.append({"type": "text", "text": text_prompt})

                self.sessions[session_id]["conversation"].append({"role": "user", "content": content})
                
                inputs = self.processor.apply_chat_template(
                    self.sessions[session_id]["conversation"],
                    padding=True,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cpu" if self.is_npu else "cuda:0")

                generation_config = GenerationConfig.from_pretrained("./src/")

                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_length=4096,
                    past_key_values=self.sessions[session_id]["past_key_values"],
                    use_cache=True,
                    return_dict_in_generate=True,
                )

                with get_image_features_callback(self.model, on_image_processing_done):
                    outputs = self.model.generate(generation_config=generation_config, **generation_kwargs)
                    if hasattr(outputs, "past_key_values"):
                        self.sessions[session_id]["past_key_values"] = outputs.past_key_values

            except StopIteration:
                logging.info(f"[{session_id}] - Generation task aborted by user.")

            except Exception as e:
                logging.error(f"[{session_id}] - Error in task thread: {e}\n {traceback.format_exc()}")

            finally:
                streamer.end()

        task_thread = Thread(target=task)

        def streamer_loop():
            answer = ""
            is_aborted = False
            try:
                for token in streamer:
                    answer += token
                    on_token(token)

            except Exception as e:
                logging.warning(f"[{session_id}] - Streamer loop interrupted: {e}")
                is_aborted = True

            finally:
                task_thread.join()
                is_aborted = is_aborted or stop_event.is_set()

                if not is_aborted:
                    assistant_content = [{"type": "text", "text": answer}]
                    self.sessions[session_id]["conversation"].append(
                        {"role": "assistant", "content": assistant_content}
                    )
                # print(f"[DEBUG] model answer: {answer}")

                on_end(is_aborted)

        streamer_thread = Thread(target=streamer_loop)

        self.sessions[session_id].update(
            {
                "task_thread": task_thread,
                "streamer_thread": streamer_thread,
                "stop_event": stop_event,
            }
        )

        task_thread.start()
        streamer_thread.start()

    def abort_generation(self, session_id: str):
        if session_id in self.sessions and "stop_event" in self.sessions[session_id]:
            logging.info(f"[{session_id}] - Aborting generation.")
            self.sessions[session_id]["stop_event"].set()

        else:
            logging.warning(f"[{session_id}] - No active generation to abort")

    def wait_for_generation(self, session_id: str, timeout: Optional[float] = None):
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]
        task_thread = session.get("task_thread")
        streamer_thread = session.get("streamer_thread")

        if task_thread is not None and task_thread.is_alive():
            task_thread.join(timeout=timeout)

        if streamer_thread is not None and streamer_thread.is_alive():
            streamer_thread.join(timeout=timeout)
