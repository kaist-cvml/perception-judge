"""
Unified Async Client for concurrent API requests (supports OpenAI and Qwen-compatible APIs).
"""
import asyncio
import base64
import io
import os
import time
from typing import Optional

import httpx
from PIL import Image


class AsyncClient:
    GPT5_MODELS = ["gpt-5"]
    OPENAI_MODELS = ["gpt-4o"]
    QWEN_MODELS = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ]

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 8192,
        timeout: float = 180.0,
        max_concurrent: int = 10,
        max_image_size: int = 1024,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_image_size = max_image_size

        if model in self.GPT5_MODELS:
            self.endpoint = "https://api.openai.com/v1/responses"
            self.use_auth = True
        elif model in self.OPENAI_MODELS:
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.use_auth = True
        elif base_url:
            # Any Qwen-compatible vLLM server
            self.endpoint = f"{base_url}/v1/chat/completions"
            self.use_auth = False
        else:
            raise ValueError(
                f"Unsupported model: {model}\n"
                f"For OpenAI models use one of: {self.GPT5_MODELS + self.OPENAI_MODELS}\n"
                f"For custom vLLM servers, provide --base_url"
            )

    async def _encode_image(self, image_path: str) -> Optional[tuple]:
        encode_start = time.time()
        try:
            image_name = os.path.basename(image_path)
            original_size = os.path.getsize(image_path)

            img = Image.open(image_path)
            original_width, original_height = img.size
            max_dimension = max(original_width, original_height)

            if max_dimension > self.max_image_size:
                scale = self.max_image_size / max_dimension
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"🔄 Resized: {image_name} | {original_width}x{original_height} → {new_width}x{new_height}")

            image_format = os.path.splitext(image_path)[1][1:].lower()
            if image_format == 'jpg':
                image_format = 'jpeg'

            buffer = io.BytesIO()
            if image_format in ['jpeg', 'jpg']:
                img.convert('RGB').save(buffer, format='JPEG', quality=90, optimize=True)
            elif image_format == 'png':
                img.save(buffer, format='PNG', optimize=True)
            else:
                img.save(buffer, format=image_format.upper())

            buffer.seek(0)
            image_bytes = buffer.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            optimized_size = len(image_bytes)
            file_size_mb = optimized_size / (1024 * 1024)
            reduction = 100 * (1 - optimized_size / original_size) if original_size > 0 else 0
            encode_time = time.time() - encode_start
            print(
                f"\t📦 Encoded: {image_name} | "
                f"{original_size/1024:.1f}KB → {optimized_size/1024:.1f}KB (-{reduction:.0f}%) | "
                f"{encode_time:.3f}s"
            )
            return image_format, image_base64, file_size_mb, encode_time

        except Exception as e:
            print(f"❌ Encode error {os.path.basename(image_path)}: {e}")
            return None

    async def call_api(self, image_path: str, prompt: str, max_tokens: Optional[int] = None) -> Optional[str]:
        total_start = time.time()
        image_name = os.path.basename(image_path)

        semaphore_start = time.time()
        async with self.semaphore:
            semaphore_wait = time.time() - semaphore_start
            if semaphore_wait > 0.1:
                print(f"⏳ Semaphore wait: {image_name} | {semaphore_wait:.2f}s")

            result = await self._encode_image(image_path)
            if not result:
                return None

            image_format, image_base64, file_size_mb, encode_time = result

            if self.model in self.GPT5_MODELS:
                payload = {
                    "model": self.model,
                    "input": [{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:image/{image_format};base64,{image_base64}"}
                        ],
                    }],
                }
            else:
                payload = {
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_base64}"}}
                        ]
                    }],
                    "max_tokens": max_tokens or self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }

            try:
                print(f"\t🚀 Request: {image_name} | prompt: {len(prompt)} chars | image: {file_size_mb:.2f}MB")
                request_start = time.time()

                headers = {"Content-Type": "application/json"}
                if self.use_auth:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    response_json = response.json()

                    if self.model in self.GPT5_MODELS:
                        content = None
                        for output_item in response_json.get('output', []):
                            if output_item.get('type') == 'message':
                                for content_item in output_item.get('content', []):
                                    if content_item.get('type') == 'output_text':
                                        content = content_item.get('text', '')
                                        break
                                if content:
                                    break
                        if not content:
                            content = str(response_json)
                    else:
                        content = response_json['choices'][0]['message']['content']

                    request_time = time.time() - request_start
                    total_time = time.time() - total_start
                    print(f"\t✅ Response: {image_name} | req: {request_time:.2f}s | total: {total_time:.2f}s | {len(content)} chars")
                    return content

            except httpx.HTTPError as e:
                print(f"❌ HTTP error: {image_name} | {e}")
                return None
            except KeyError:
                print(f"❌ Response format error: {image_name} | {response.json()}")
                return None
            except Exception as e:
                print(f"❌ API error: {image_name} | {e}")
                return None


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
