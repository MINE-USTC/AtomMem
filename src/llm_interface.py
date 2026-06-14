# src/llm_interface.py
# LLM API Interface

import json
import time
from typing import Dict, Any, Optional
from openai import OpenAI
import httpx
import config


class LLMInterface:
    """LLM API interface for calling Qwen via DashScope compatible mode."""
    
    def __init__(self):
        """Initialize LLM client."""
        # Build a custom httpx client:
        #   - timeout: prevents the request from hanging indefinitely
        #   - trust_env=False: bypasses system proxy (DashScope is a Chinese
        #     service reachable directly; a system proxy may block it)
        _http_client_kwargs = dict(
            timeout=httpx.Timeout(config.LLM_REQUEST_TIMEOUT, connect=10.0)
        )
        if config.LLM_BYPASS_PROXY:
            _http_client_kwargs["trust_env"] = False

        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.API_BASE,
            http_client=httpx.Client(**_http_client_kwargs)
        )
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS
        
        # Aggregate statistics
        self.total_tokens = 0
        self.total_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def call(self, 
             system_prompt: str, 
             user_prompt: str,
             response_format: Optional[str] = "json",
             call_type: str = "unknown") -> Dict[str, Any]:
        """
        Call LLM API.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            response_format: Expected response format ("json" or "text")
            call_type: Optional caller label kept for backwards-compatible call sites.
            
        Returns:
            Parsed response from LLM
        """
        _ = call_type
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Build API call kwargs
            api_kwargs = dict(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            # Request strict JSON output from the API when JSON is expected,
            # which is essential for Qwen models via DashScope compatible mode.
            if response_format == "json":
                api_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**api_kwargs)
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.total_calls += 1
                self.prompt_tokens += response.usage.prompt_tokens
                self.completion_tokens += response.usage.completion_tokens
                self.total_tokens += response.usage.total_tokens
            
            content = response.choices[0].message.content
            
            if response_format == "json":
                # Try to parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Fallback: extract JSON from markdown code blocks
                    import re
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(1))
                    # Last resort: return raw content wrapped so callers detect the failure
                    return {"raw_content": content}
            else:
                return {"content": content}
                
        except Exception as e:
            return {"error": str(e)}
    
    def call_with_retry(self,
                       system_prompt: str,
                       user_prompt: str,
                       response_format: Optional[str] = "json",
                       max_retries: int = 3,
                       call_type: str = "unknown") -> Dict[str, Any]:
        """
        Call LLM API with retry mechanism and exponential backoff.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            response_format: Expected response format
            max_retries: Maximum number of retries
            call_type: Optional caller label kept for backwards-compatible call sites.
            
        Returns:
            Parsed response from LLM
        """
        for attempt in range(max_retries):
            result = self.call(system_prompt, user_prompt, response_format, call_type=call_type)
            if "error" not in result:
                return result
            # Print the error so it is visible rather than silently swallowed
            err_msg = result.get("error", "unknown error")
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1 s, 2 s, 4 s …
                print(f"  [LLM] Attempt {attempt + 1}/{max_retries} failed: {err_msg}. "
                      f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [LLM] Attempt {attempt + 1}/{max_retries} failed: {err_msg}. "
                      f"No more retries.")
        return {"error": "Max retries reached"}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get LLM usage statistics.
        
        Returns:
            Dictionary with aggregate token/call counts.
        """
        return {
            "total_tokens": self.total_tokens,
            "total_tokens_k": round(self.total_tokens / 1000, 2),
            "calls": self.total_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_tokens = 0
        self.total_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        print(f"\n{'='*60}")
        print("LLM USAGE STATISTICS")
        print(f"{'='*60}")
        print(f"Total Tokens:       {stats['total_tokens']:>10,} ({stats['total_tokens_k']:.2f}k)")
        print(f"  - Prompt Tokens:  {stats['prompt_tokens']:>10,}")
        print(f"  - Completion:     {stats['completion_tokens']:>10,}")
        print(f"Total API Calls:    {stats['calls']:>10,}")
        print(f"{'='*60}\n")
